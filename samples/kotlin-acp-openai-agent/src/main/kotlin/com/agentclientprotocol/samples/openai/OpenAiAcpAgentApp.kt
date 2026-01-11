package com.agentclientprotocol.samples.openai

import com.agentclientprotocol.agent.*
import com.agentclientprotocol.common.Event
import com.agentclientprotocol.common.SessionCreationParameters
import com.agentclientprotocol.model.*
import com.agentclientprotocol.protocol.Protocol
import com.agentclientprotocol.transport.StdioTransport
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import io.github.oshai.kotlinlogging.KotlinLogging
import io.modelcontextprotocol.client.McpClient
import io.modelcontextprotocol.client.McpSyncClient
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport
import io.modelcontextprotocol.client.transport.ServerParameters
import io.modelcontextprotocol.client.transport.StdioClientTransport
import io.modelcontextprotocol.json.jackson.JacksonMcpJsonMapper
import io.modelcontextprotocol.spec.McpSchema
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.FlowCollector
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.io.asSink
import kotlinx.io.asSource
import kotlinx.io.buffered
import kotlinx.serialization.json.*
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.Message
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.chat.prompt.Prompt
import org.springframework.ai.model.tool.DefaultToolCallingManager
import org.springframework.ai.model.tool.ToolExecutionResult
import org.springframework.ai.openai.OpenAiChatModel
import org.springframework.ai.openai.OpenAiChatOptions
import org.springframework.ai.openai.api.OpenAiApi
import org.springframework.ai.tool.ToolCallback
import org.springframework.ai.tool.definition.DefaultToolDefinition
import org.springframework.ai.tool.resolution.StaticToolCallbackResolver
import org.springframework.util.CollectionUtils
import java.net.URI
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import kotlin.coroutines.CoroutineContext

private val logger = KotlinLogging.logger {}
private val toolJson = Json { ignoreUnknownKeys = true }

fun main(): Unit = runBlocking {
    val transport = StdioTransport(
        parentScope = this,
        ioDispatcher = Dispatchers.IO,
        input = System.`in`.asSource().buffered(),
        output = System.out.asSink().buffered()
    )
    val protocol = Protocol(this, transport)
    val client = OpenAiClient.fromEnv()

    Agent(
        protocol = protocol,
        agentSupport = OpenAiAgentSupport(client)
    )

    protocol.start()
}

internal class OpenAiAgentSupport(
    private val client: OpenAiClient
) : AgentSupport {
    override suspend fun initialize(clientInfo: com.agentclientprotocol.client.ClientInfo): AgentInfo {
        return AgentInfo(
            protocolVersion = LATEST_PROTOCOL_VERSION,
            capabilities = AgentCapabilities(
                loadSession = true,
                promptCapabilities = PromptCapabilities(embeddedContext = true),
                mcpCapabilities = McpCapabilities(http = true, sse = false)
            )
        )
    }

    override suspend fun createSession(sessionParameters: SessionCreationParameters): AgentSession {
        val sessionId = SessionId("session-${System.currentTimeMillis()}")
        return OpenAiAgentSession(sessionId, client, sessionParameters)
    }

    override suspend fun loadSession(
        sessionId: SessionId,
        sessionParameters: SessionCreationParameters
    ): AgentSession {
        return OpenAiAgentSession(sessionId, client, sessionParameters)
    }
}

class OpenAiAgentSession(
    override val sessionId: SessionId,
    val client: OpenAiClient,
    val sessionParameters: SessionCreationParameters
) : AgentSession {

    private var fileOpsContext: String = ""
    private var terminalOpsContext: String = ""

    private var greeted = false
    private val toolCallingManager: DefaultToolCallingManager;
    private val mcpResources = buildMcpResources(sessionParameters.mcpServers)
    private val messageHistory = mutableListOf<Message>().apply {
        if (client.systemPrompt.isNotBlank()) {
            add(SystemMessage(client.systemPrompt))
        }
    }

    init {
        if (mcpResources.toolCallbacks.isNotEmpty()) {
            toolCallingManager = DefaultToolCallingManager.builder()
                .toolCallbackResolver(StaticToolCallbackResolver(mcpResources.toolCallbacks))
                .build()
        } else {
            toolCallingManager = DefaultToolCallingManager.builder()
                .build()
        }
    }


    override suspend fun prompt(
        content: List<ContentBlock>,
        _meta: kotlinx.serialization.json.JsonElement?
    ): Flow<Event> = flow {
        val context = currentCoroutineContext()
        val clientCapabilities = context.clientInfo.capabilities
        val fileSystemCapabilities = clientCapabilities.fs
        val terminalCapabilities = clientCapabilities.terminal
        logInfo("received prompt")

        if (!greeted) {
            greeted = true
//            val greeting = buildGreeting(fileSystemCapabilities != null, terminalCapabilities)
//            if (greeting.isNotBlank()) {
//                emit(
//                    Event.SessionUpdateEvent(
//                        SessionUpdate.AgentMessageChunk(ContentBlock.Text(greeting))
//                    )
//                )
//            }
        }

        val userText = content.filterIsInstance<ContentBlock.Text>()
            .joinToString("\n") { it.text }
            .trim()

        val (parsedOperations, fileOpResults, terminalOpResults) = executeOps(
            fileSystemCapabilities,
            terminalCapabilities,
            userText,
            context
        )

        val resourceContext = buildResourceContext(content)
        val fileOpContext = buildFileOperationContext(fileOpResults)
        val terminalOpContext = buildTerminalOperationContext(terminalOpResults)

        val promptText = listOfNotNull(
            parsedOperations.remainingText.takeIf { it.isNotBlank() } ?: "(empty)",
            resourceContext,
            fileOpsContext + fileOpContext,
            terminalOpsContext + terminalOpContext
        ).joinToString("\n\n")

        messageHistory.add(UserMessage(promptText))
        val completionResult = withContext(Dispatchers.IO) {
            client.createChatCompletion(
                messageHistory.toList(),
                mcpResources.toolCallbacks,
                toolCallingManager
            )
        }
        messageHistory.clear()
        messageHistory.addAll(completionResult.updatedHistory)


        val (p, fileOpResultsAgent, terminalOpResultsAgent) = executeOps(
            fileSystemCapabilities,
            terminalCapabilities,
            completionResult.content,
            context
        )

        fileOpsContext = buildFileOperationContext(fileOpResultsAgent) ?: ""
        terminalOpsContext = buildTerminalOperationContext(terminalOpResultsAgent) ?: ""

        emit(
            Event.SessionUpdateEvent(
                SessionUpdate.AgentMessageChunk(ContentBlock.Text(completionResult.content))
            )
        )
        emit(Event.PromptResponseEvent(PromptResponse(StopReason.END_TURN)))
    }
        .flowOn(Dispatchers.IO)

    private suspend fun FlowCollector<Event>.executeOps(
        fileSystemCapabilities: FileSystemCapability?,
        terminalCapabilities: Boolean,
        userText: String,
        context: CoroutineContext
    ): Triple<TerminalOperationParseResult, List<FileOperationResult>, List<TerminalOperationResult>> {
        val fileOpsEnabled =
            fileSystemCapabilities?.readTextFile == true || fileSystemCapabilities?.writeTextFile == true
        val terminalOpsEnabled = terminalCapabilities
        val parsedOperations = parseOperations(userText, fileOpsEnabled, terminalOpsEnabled)

        logInfo(fileOpsEnabled.toString())

        val fileOpResults = if (parsedOperations.fileRequests.isNotEmpty()) {
            executeFileOperations(context, parsedOperations.fileRequests)
        } else {
            emptyList()
        }
        val terminalOpResults = if (parsedOperations.terminalRequests.isNotEmpty()) {
            executeTerminalOperations(context, parsedOperations.terminalRequests)
        } else {
            emptyList()
        }
        return Triple(parsedOperations, fileOpResults, terminalOpResults)
    }

    override suspend fun cancel() {
        mcpResources.close()
    }

    private suspend fun kotlinx.coroutines.flow.FlowCollector<Event>.executeFileOperations(
        context: kotlin.coroutines.CoroutineContext,
        requests: List<FileOperationRequest>
    ): List<FileOperationResult> {
        val results = mutableListOf<FileOperationResult>()
        val clientOperations = context.client
        val fsCapabilities = context.clientInfo.capabilities.fs

        requests.forEachIndexed { index, request ->
            val toolCallId = ToolCallId("fs-${System.currentTimeMillis()}-$index")
            val resolvedPath = resolvePath(sessionParameters.cwd, request.path)
            val toolKind = if (request.kind == FileOperationKind.READ) ToolKind.READ else ToolKind.EDIT

            emitToolUpdate(
                toolCallId,
                title = request.describeTitle(),
                kind = toolKind,
                status = ToolCallStatus.PENDING,
                path = resolvedPath
            )
            emitToolUpdate(
                toolCallId,
                status = ToolCallStatus.IN_PROGRESS
            )

            val result = try {
                when (request.kind) {
                    FileOperationKind.READ -> {
                        if (fsCapabilities?.readTextFile != true) {
                            FileOperationResult(request, resolvedPath, null, "fs.readTextFile not supported by client")
                        } else {
                            val response = clientOperations.fsReadTextFile(
                                resolvedPath,
                                line = request.line,
                                limit = request.limit
                            )
                            FileOperationResult(request, resolvedPath, response.content, null)
                        }
                    }

                    FileOperationKind.WRITE -> {
                        if (fsCapabilities?.writeTextFile != true) {
                            FileOperationResult(request, resolvedPath, null, "fs.writeTextFile not supported by client")
                        } else {
                            clientOperations.fsWriteTextFile(resolvedPath, request.content.orEmpty())
                            FileOperationResult(request, resolvedPath, null, null)
                        }
                    }
                }
            } catch (ex: Exception) {
                FileOperationResult(request, resolvedPath, null, ex.message ?: "File operation failed")
            }

            val toolContent = buildToolCallContent(result)
            emitToolUpdate(
                toolCallId,
                status = if (result.error == null) ToolCallStatus.COMPLETED else ToolCallStatus.FAILED,
                content = toolContent
            )

            results.add(result)
        }

        return results
    }

    private suspend fun kotlinx.coroutines.flow.FlowCollector<Event>.executeTerminalOperations(
        context: kotlin.coroutines.CoroutineContext,
        requests: List<TerminalOperationRequest>
    ): List<TerminalOperationResult> {
        val results = mutableListOf<TerminalOperationResult>()
        val clientOperations = context.client
        val terminalEnabled = context.clientInfo.capabilities.terminal

        requests.forEachIndexed { index, request ->
            val toolCallId = ToolCallId("term-${System.currentTimeMillis()}-$index")
            val toolKind = when (request.kind) {
                TerminalOperationKind.CREATE -> ToolKind.EXECUTE
                TerminalOperationKind.OUTPUT -> ToolKind.READ
                TerminalOperationKind.WAIT -> ToolKind.OTHER
                TerminalOperationKind.KILL -> ToolKind.EXECUTE
                TerminalOperationKind.RELEASE -> ToolKind.OTHER
            }

            emitToolUpdate(
                toolCallId,
                title = request.describeTitle(),
                kind = toolKind,
                status = ToolCallStatus.PENDING
            )
            emitToolUpdate(
                toolCallId,
                status = ToolCallStatus.IN_PROGRESS
            )

            val result = try {
                if (!terminalEnabled) {
                    TerminalOperationResult(request, null, null, null, "terminal capability not supported by client")
                } else {
                    when (request.kind) {
                        TerminalOperationKind.CREATE -> {
                            val spec = request.createSpec
                            if (spec == null) {
                                TerminalOperationResult(request, null, null, null, "Missing terminal create payload")
                            } else {
                                val response = clientOperations.terminalCreate(
                                    spec.command,
                                    spec.args,
                                    spec.cwd,
                                    spec.env,
                                    spec.outputByteLimit
                                )
                                TerminalOperationResult(request, response.terminalId, null, null, null)
                            }
                        }

                        TerminalOperationKind.OUTPUT -> {
                            val terminalId = request.terminalId.orEmpty()
                            val response = clientOperations.terminalOutput(terminalId)
                            TerminalOperationResult(request, terminalId, response.output, null, null)
                        }

                        TerminalOperationKind.WAIT -> {
                            val terminalId = request.terminalId.orEmpty()
                            val response = clientOperations.terminalWaitForExit(terminalId)
                            TerminalOperationResult(request, terminalId, null, response.exitCode, null)
                        }

                        TerminalOperationKind.KILL -> {
                            val terminalId = request.terminalId.orEmpty()
                            clientOperations.terminalKill(terminalId)
                            TerminalOperationResult(request, terminalId, null, null, null)
                        }

                        TerminalOperationKind.RELEASE -> {
                            val terminalId = request.terminalId.orEmpty()
                            clientOperations.terminalRelease(terminalId)
                            TerminalOperationResult(request, terminalId, null, null, null)
                        }
                    }
                }
            } catch (ex: Exception) {
                TerminalOperationResult(request, request.terminalId, null, null, ex.message ?: "Terminal operation failed")
            }

            val toolContent = buildTerminalToolCallContent(result)
            emitToolUpdate(
                toolCallId,
                status = if (result.error == null) ToolCallStatus.COMPLETED else ToolCallStatus.FAILED,
                content = toolContent
            )

            results.add(result)
        }

        return results
    }

    private fun buildTerminalToolCallContent(result: TerminalOperationResult): List<ToolCallContent> {
        val message = when {
            result.error != null -> "Terminal operation failed: ${result.error}"
            result.request.kind == TerminalOperationKind.CREATE -> "Created terminal: ${result.terminalId}"
            result.request.kind == TerminalOperationKind.OUTPUT -> "Terminal output (${result.terminalId}):\n${truncateForPrompt(result.output.orEmpty())}"
            result.request.kind == TerminalOperationKind.WAIT -> "Terminal exit (${result.terminalId}): ${result.exitCode}"
            result.request.kind == TerminalOperationKind.KILL -> "Killed terminal: ${result.terminalId}"
            result.request.kind == TerminalOperationKind.RELEASE -> "Released terminal: ${result.terminalId}"
            else -> "Terminal operation completed."
        }
        val responseType = when {
            result.error != null -> "term.error"
            result.request.kind == TerminalOperationKind.CREATE -> "term.create"
            result.request.kind == TerminalOperationKind.OUTPUT -> "term.output"
            result.request.kind == TerminalOperationKind.WAIT -> "term.wait"
            result.request.kind == TerminalOperationKind.KILL -> "term.kill"
            result.request.kind == TerminalOperationKind.RELEASE -> "term.release"
            else -> "term.unknown"
        }
        val wrapped = wrapToolResponse(responseType, message)
        return listOf(ToolCallContent.Content(ContentBlock.Text(wrapped)))
    }

    private fun buildToolCallContent(result: FileOperationResult): List<ToolCallContent> {
        val message = when {
            result.error != null -> "File operation failed: ${result.error}"
            result.request.kind == FileOperationKind.READ -> {
                val preview = truncateForPrompt(result.content.orEmpty())
                "Read ${result.path}:\n$preview"
            }

            else -> "Wrote ${result.path} (${result.request.content?.length ?: 0} chars)."
        }
        val responseType = when {
            result.error != null -> "fs.error"
            result.request.kind == FileOperationKind.READ -> "fs.read"
            else -> "fs.write"
        }
        val wrapped = wrapToolResponse(responseType, message)
        return listOf(ToolCallContent.Content(ContentBlock.Text(wrapped)))
    }

    private suspend fun kotlinx.coroutines.flow.FlowCollector<Event>.emitToolUpdate(
        toolCallId: ToolCallId,
        title: String? = null,
        kind: ToolKind? = null,
        status: ToolCallStatus? = null,
        path: String? = null,
        content: List<ToolCallContent>? = null
    ) {
        emit(
            Event.SessionUpdateEvent(
                SessionUpdate.ToolCallUpdate(
                    toolCallId = toolCallId,
                    title = title,
                    kind = kind,
                    status = status,
                    locations = path?.let { listOf(ToolCallLocation(it)) },
                    content = content
                )
            )
        )
    }
}

private fun logInfo(fileOpsEnabled: String) {
    Files.write(
        Paths.get("/Users/hayde/IdeaProjects/multi_agent_ide_parent/libs/acp-kotlin/hello.txt"),
        fileOpsEnabled.toByteArray(),
        StandardOpenOption.APPEND
    )
}

private enum class FileOperationKind {
    READ,
    WRITE
}

private data class FileOperationRequest(
    val kind: FileOperationKind,
    val path: String,
    val content: String? = null,
    val line: UInt? = null,
    val limit: UInt? = null
) {
    fun describeTitle(): String = when (kind) {
        FileOperationKind.READ -> "Read file"
        FileOperationKind.WRITE -> "Write file"
    }
}

private data class FileOperationResult(
    val request: FileOperationRequest,
    val path: String,
    val content: String?,
    val error: String?
)

private data class TerminalOperationParseResult(
    val remainingText: String,
    val fileRequests: List<FileOperationRequest>,
    val terminalRequests: List<TerminalOperationRequest>
)

private data class FsWriteParseResult(
    val path: String?,
    val content: String,
    val nextIndex: Int
)

private data class FsReadParseResult(
    val path: String?,
    val line: UInt?,
    val limit: UInt?,
    val nextIndex: Int
)

private fun parseOperations(
    text: String,
    fileOpsEnabled: Boolean,
    terminalOpsEnabled: Boolean
): TerminalOperationParseResult {
    if (text.isBlank()) return TerminalOperationParseResult(text, emptyList(), emptyList())

    val requests = mutableListOf<FileOperationRequest>()
    val terminalRequests = mutableListOf<TerminalOperationRequest>()
    val remaining = StringBuilder()
    val lines = text.split('\n')
    var index = 0

    while (index < lines.size) {
        val line = lines[index]
        val trimmed = line.trim()

        if (fileOpsEnabled) {
            if (trimmed.startsWith("<fs.read")) {
                val result = parseFsReadXml(lines, index)
                if (!result.path.isNullOrBlank()) {
                    requests.add(
                        FileOperationRequest(
                            FileOperationKind.READ,
                            result.path.trim(),
                            line = result.line,
                            limit = result.limit
                        )
                    )
                }
                index = result.nextIndex
                continue
            }

            if (trimmed.startsWith("<fs.write")) {
                val result = parseFsWriteXml(lines, index)
                if (!result.path.isNullOrBlank()) {
                    requests.add(FileOperationRequest(FileOperationKind.WRITE, result.path.trim(), result.content))
                }
                index = result.nextIndex
                continue
            }

            if (trimmed.startsWith("/fs.read ")) {
                val path = trimmed.removePrefix("/fs.read ").trim()
                if (path.isNotBlank()) {
                    requests.add(FileOperationRequest(FileOperationKind.READ, path))
                }
                index += 1
                continue
            }

            if (trimmed.startsWith("/fs.write ")) {
                val path = trimmed.removePrefix("/fs.write ").trim()
                index += 1
                val contentLines = mutableListOf<String>()
                while (index < lines.size) {
                    val inner = lines[index]
                    val innerTrim = inner.trim()
                    if (innerTrim.startsWith("/fs.end") || innerTrim.startsWith("/fs.read ") || innerTrim.startsWith("/fs.write ")) {
                        break
                    }
                    contentLines.add(inner)
                    index += 1
                }
                if (index < lines.size && lines[index].trim().startsWith("/fs.end")) {
                    index += 1
                }
                if (path.isNotBlank()) {
                    requests.add(FileOperationRequest(FileOperationKind.WRITE, path, contentLines.joinToString("\n")))
                }
                continue
            }
        }

        if (terminalOpsEnabled) {
            if (trimmed.startsWith("<term.create")) {
                val (payload, nextIndex) = parseXmlBlock(lines, index, "term.create")
                val spec = parseTerminalCreateSpec(payload.orEmpty())
                if (spec != null) {
                    terminalRequests.add(TerminalOperationRequest.create(spec))
                }
                index = nextIndex
                continue
            }

            val terminalXmlId = when {
                trimmed.startsWith("<term.output") -> parseXmlBlock(lines, index, "term.output")
                trimmed.startsWith("<term.wait") -> parseXmlBlock(lines, index, "term.wait")
                trimmed.startsWith("<term.kill") -> parseXmlBlock(lines, index, "term.kill")
                trimmed.startsWith("<term.release") -> parseXmlBlock(lines, index, "term.release")
                else -> null
            }

            if (terminalXmlId != null) {
                val terminalId = terminalXmlId.first.orEmpty().trim()
                if (terminalId.isNotBlank()) {
                    val request = when {
                        trimmed.startsWith("<term.output") -> TerminalOperationRequest.output(terminalId)
                        trimmed.startsWith("<term.wait") -> TerminalOperationRequest.waitForExit(terminalId)
                        trimmed.startsWith("<term.kill") -> TerminalOperationRequest.kill(terminalId)
                        else -> TerminalOperationRequest.release(terminalId)
                    }
                    terminalRequests.add(request)
                }
                index = terminalXmlId.second
                continue
            }

            if (trimmed.startsWith("/term.create")) {
                val (spec, nextIndex) = parseTerminalCreate(lines, index, trimmed)
                if (spec != null) {
                    terminalRequests.add(TerminalOperationRequest.create(spec))
                }
                index = nextIndex
                continue
            }

            val terminalId = when {
                trimmed.startsWith("/term.output ") -> trimmed.removePrefix("/term.output ").trim()
                trimmed.startsWith("/term.wait ") -> trimmed.removePrefix("/term.wait ").trim()
                trimmed.startsWith("/term.kill ") -> trimmed.removePrefix("/term.kill ").trim()
                trimmed.startsWith("/term.release ") -> trimmed.removePrefix("/term.release ").trim()
                else -> null
            }

            if (terminalId != null && terminalId.isNotBlank()) {
                val request = when {
                    trimmed.startsWith("/term.output ") -> TerminalOperationRequest.output(terminalId)
                    trimmed.startsWith("/term.wait ") -> TerminalOperationRequest.waitForExit(terminalId)
                    trimmed.startsWith("/term.kill ") -> TerminalOperationRequest.kill(terminalId)
                    else -> TerminalOperationRequest.release(terminalId)
                }
                logInfo("Adding terminal")
                terminalRequests.add(request)
                index += 1
                continue
            }
        }

        remaining.append(line)
        remaining.append('\n')
        index += 1
    }

    return TerminalOperationParseResult(remaining.toString().trim(), requests, terminalRequests)
}

private fun parseFsReadXml(lines: List<String>, startIndex: Int): FsReadParseResult {
    val lineTrim = lines[startIndex].trim()
    val attrPath = parseXmlAttribute(lineTrim, "path")
    val attrLine = parseXmlUIntAttribute(lineTrim, "line")
    val attrLimit = parseXmlUIntAttribute(lineTrim, "limit")
    val (contentPath, nextIndex) = parseXmlBlock(lines, startIndex, "fs.read")
    val path = attrPath ?: contentPath
    return FsReadParseResult(path, attrLine, attrLimit, nextIndex)
}

private fun parseFsWriteXml(lines: List<String>, startIndex: Int): FsWriteParseResult {
    val lineTrim = lines[startIndex].trim()
    val path = parseXmlAttribute(lineTrim, "path")
    val (content, nextIndex) = parseXmlBlock(lines, startIndex, "fs.write")
    return FsWriteParseResult(path, content.orEmpty(), nextIndex)
}

private fun parseXmlBlock(
    lines: List<String>,
    startIndex: Int,
    tagName: String
): Pair<String?, Int> {
    val startLine = lines[startIndex].trim()
    val startTag = "<$tagName"
    val closeTag = "</$tagName>"
    if (!startLine.startsWith(startTag)) return Pair(null, startIndex + 1)

    val inlineContent = startLine.substringAfter(">", "")
    if (startLine.contains(closeTag)) {
        val content = inlineContent.substringBefore(closeTag)
        return Pair(content, startIndex + 1)
    }

    val contentLines = mutableListOf<String>()
    if (inlineContent.isNotBlank()) {
        contentLines.add(inlineContent)
    }
    var index = startIndex + 1
    while (index < lines.size) {
        val line = lines[index]
        val lineTrim = line.trim()
        if (lineTrim.contains(closeTag)) {
            val beforeClose = line.substringBefore(closeTag)
            if (beforeClose.isNotBlank()) {
                contentLines.add(beforeClose)
            }
            index += 1
            break
        }
        contentLines.add(line)
        index += 1
    }
    return Pair(contentLines.joinToString("\n").trimEnd(), index)
}

private fun parseXmlAttribute(line: String, name: String): String? {
    val regex = Regex("""\b${Regex.escape(name)}\s*=\s*("([^"]*)"|'([^']*)')""")
    val match = regex.find(line) ?: return null
    return match.groups[2]?.value ?: match.groups[3]?.value
}

private fun parseXmlUIntAttribute(line: String, name: String): UInt? {
    return parseXmlAttribute(line, name)?.toUIntOrNull()
}

private fun buildResourceContext(content: List<ContentBlock>): String? {
    val blocks = mutableListOf<String>()
    content.forEach { block ->
        when (block) {
            is ContentBlock.Resource -> {
                when (val resource = block.resource) {
                    is com.agentclientprotocol.model.EmbeddedResourceResource.TextResourceContents -> {
                        val preview = truncateForPrompt(resource.text)
                        blocks.add("Embedded resource (${resource.uri}):\n$preview")
                    }

                    is com.agentclientprotocol.model.EmbeddedResourceResource.BlobResourceContents -> {
                        blocks.add("Embedded binary resource (${resource.uri}, mimeType=${resource.mimeType ?: "unknown"})")
                    }
                }
            }

            is ContentBlock.ResourceLink -> {
                val description = block.description?.let { " - $it" } ?: ""
                blocks.add("Resource link (${block.name}): ${block.uri}$description")
            }

            is ContentBlock.Image -> blocks.add("Image content provided (${block.mimeType}).")
            is ContentBlock.Audio -> blocks.add("Audio content provided (${block.mimeType}).")
            else -> Unit
        }
    }

    return blocks.takeIf { it.isNotEmpty() }?.joinToString("\n\n")
}

private fun buildFileOperationContext(results: List<FileOperationResult>): String? {
    if (results.isEmpty()) return null
    val lines = results.map { result ->
        val message = when {
            result.error != null -> "File operation failed (${result.request.kind} ${result.path}): ${result.error}"
            result.request.kind == FileOperationKind.READ -> {
                val preview = truncateForPrompt(result.content.orEmpty())
                "Read ${result.path}:\n$preview"
            }

            else -> "Wrote ${result.path} (${result.request.content?.length ?: 0} chars)."
        }
        val responseType = when {
            result.error != null -> "fs.error"
            result.request.kind == FileOperationKind.READ -> "fs.read"
            else -> "fs.write"
        }
        wrapToolResponse(responseType, message)
    }
    return "File operations:\n${lines.joinToString("\n\n")}"
}

private fun buildTerminalOperationContext(results: List<TerminalOperationResult>): String? {
    if (results.isEmpty()) return null
    val lines = results.map { result ->
        val message = when {
            result.error != null -> "Terminal operation failed (${result.request.describeTitle()}): ${result.error}"
            result.request.kind == TerminalOperationKind.CREATE -> "Created terminal ${result.terminalId}"
            result.request.kind == TerminalOperationKind.OUTPUT -> "Output from ${result.terminalId}:\n${truncateForPrompt(result.output.orEmpty())}"
            result.request.kind == TerminalOperationKind.WAIT -> "Exit code for ${result.terminalId}: ${result.exitCode}"
            result.request.kind == TerminalOperationKind.KILL -> "Killed terminal ${result.terminalId}"
            result.request.kind == TerminalOperationKind.RELEASE -> "Released terminal ${result.terminalId}"
            else -> "Terminal operation completed."
        }
        val responseType = when {
            result.error != null -> "term.error"
            result.request.kind == TerminalOperationKind.CREATE -> "term.create"
            result.request.kind == TerminalOperationKind.OUTPUT -> "term.output"
            result.request.kind == TerminalOperationKind.WAIT -> "term.wait"
            result.request.kind == TerminalOperationKind.KILL -> "term.kill"
            result.request.kind == TerminalOperationKind.RELEASE -> "term.release"
            else -> "term.unknown"
        }
        wrapToolResponse(responseType, message)
    }
    return "Terminal operations:\n${lines.joinToString("\n\n")}"
}

private fun resolvePath(cwd: String, path: String): String {
    if (path.isBlank()) return path
    return try {
        val candidate = Paths.get(path)
        val resolved = if (candidate.isAbsolute) candidate else Paths.get(cwd).resolve(candidate)
        resolved.normalize().toString()
    } catch (ex: Exception) {
        path
    }
}

private fun truncateForPrompt(text: String, maxChars: Int = 4000): String {
    if (text.length <= maxChars) return text
    return text.take(maxChars) + "\n...[truncated ${text.length - maxChars} chars]"
}

private fun wrapToolResponse(type: String, message: String): String {
    val payload = message.trimEnd()
    return "<tool_response type=\"$type\">\n$payload\n</tool_response>"
}

private enum class TerminalOperationKind {
    CREATE,
    OUTPUT,
    WAIT,
    KILL,
    RELEASE
}

private data class TerminalCreateSpec(
    val command: String,
    val args: List<String> = emptyList(),
    val cwd: String? = null,
    val env: List<com.agentclientprotocol.model.EnvVariable> = emptyList(),
    val outputByteLimit: ULong? = null
)

private data class TerminalOperationRequest(
    val kind: TerminalOperationKind,
    val terminalId: String? = null,
    val createSpec: TerminalCreateSpec? = null
) {
    fun describeTitle(): String = when (kind) {
        TerminalOperationKind.CREATE -> "Create terminal"
        TerminalOperationKind.OUTPUT -> "Fetch terminal output"
        TerminalOperationKind.WAIT -> "Wait for terminal exit"
        TerminalOperationKind.KILL -> "Kill terminal"
        TerminalOperationKind.RELEASE -> "Release terminal"
    }

    companion object {
        fun create(spec: TerminalCreateSpec) = TerminalOperationRequest(TerminalOperationKind.CREATE, createSpec = spec)
        fun output(terminalId: String) = TerminalOperationRequest(TerminalOperationKind.OUTPUT, terminalId = terminalId)
        fun waitForExit(terminalId: String) = TerminalOperationRequest(TerminalOperationKind.WAIT, terminalId = terminalId)
        fun kill(terminalId: String) = TerminalOperationRequest(TerminalOperationKind.KILL, terminalId = terminalId)
        fun release(terminalId: String) = TerminalOperationRequest(TerminalOperationKind.RELEASE, terminalId = terminalId)
    }
}

private data class TerminalOperationResult(
    val request: TerminalOperationRequest,
    val terminalId: String?,
    val output: String?,
    val exitCode: UInt?,
    val error: String?
)

private fun parseTerminalCreate(
    lines: List<String>,
    startIndex: Int,
    trimmed: String
): Pair<TerminalCreateSpec?, Int> {
    val inline = trimmed.removePrefix("/term.create").trim()
    if (inline.isNotBlank()) {
        return Pair(parseTerminalCreateSpec(inline), startIndex + 1)
    }

    val contentLines = mutableListOf<String>()
    var index = startIndex + 1
    while (index < lines.size) {
        val line = lines[index]
        val lineTrim = line.trim()
        if (lineTrim.startsWith("/term.end") || lineTrim.startsWith("/term.create")) {
            break
        }
        contentLines.add(line)
        index += 1
    }
    if (index < lines.size && lines[index].trim().startsWith("/term.end")) {
        index += 1
    }
    val payload = contentLines.joinToString("\n").trim()
    return Pair(parseTerminalCreateSpec(payload), index)
}

private fun parseTerminalCreateSpec(raw: String): TerminalCreateSpec? {
    if (raw.isBlank()) return null
    if (raw.trimStart().startsWith("{")) {
        return parseTerminalCreateJson(raw)
    }
    val tokens = splitTokens(raw)
    if (tokens.isEmpty()) return null
    return TerminalCreateSpec(command = tokens.first(), args = tokens.drop(1))
}

private fun parseTerminalCreateJson(raw: String): TerminalCreateSpec? {
    val element = runCatching { toolJson.parseToJsonElement(raw) }.getOrNull() ?: return null
    val obj = element.jsonObject
    val command = obj["command"]?.jsonPrimitive?.contentOrNullCompat() ?: return null
    val args = obj["args"]?.jsonArray?.mapNotNull { (it as? JsonPrimitive)?.contentOrNullCompat() } ?: emptyList()
    val cwd = obj["cwd"]?.jsonPrimitive?.contentOrNullCompat()
    val outputByteLimit = obj["outputByteLimit"]?.jsonPrimitive?.toULongOrNullCompat()
    val env = parseEnv(obj["env"])
    return TerminalCreateSpec(
        command = command,
        args = args,
        cwd = cwd,
        env = env,
        outputByteLimit = outputByteLimit
    )
}

private fun parseEnv(element: kotlinx.serialization.json.JsonElement?): List<com.agentclientprotocol.model.EnvVariable> {
    if (element == null) return emptyList()
    return when (element) {
        is JsonPrimitive -> {
            val value = element.contentOrNullCompat() ?: return emptyList()
            val parts = value.split("=", limit = 2)
            if (parts.size == 2) listOf(com.agentclientprotocol.model.EnvVariable(parts[0], parts[1])) else emptyList()
        }

        else -> when (element) {
            is kotlinx.serialization.json.JsonObject -> element.map { (key, value) ->
                com.agentclientprotocol.model.EnvVariable(key, value.jsonPrimitive.content)
            }
            else -> element.jsonArray.mapNotNull { entry ->
                val value = (entry as? JsonPrimitive)?.contentOrNullCompat() ?: return@mapNotNull null
                val parts = value.split("=", limit = 2)
                if (parts.size == 2) com.agentclientprotocol.model.EnvVariable(parts[0], parts[1]) else null
            }
        }
    }
}

private fun JsonPrimitive.contentOrNullCompat(): String? = content

private fun JsonPrimitive.toULongOrNullCompat(): ULong? = content.toULongOrNull()

private fun splitTokens(text: String): List<String> {
    val tokens = mutableListOf<String>()
    val current = StringBuilder()
    var quote: Char? = null
    var index = 0
    while (index < text.length) {
        val ch = text[index]
        if (quote != null) {
            when (ch) {
                quote -> quote = null
                '\\' -> {
                    if (index + 1 < text.length) {
                        current.append(text[index + 1])
                        index += 1
                    } else {
                        current.append(ch)
                    }
                }
                else -> current.append(ch)
            }
        } else {
            when {
                ch == '"' || ch == '\'' -> quote = ch
                ch.isWhitespace() -> {
                    if (current.isNotEmpty()) {
                        tokens.add(current.toString())
                        current.setLength(0)
                    }
                }
                else -> current.append(ch)
            }
        }
        index += 1
    }
    if (current.isNotEmpty()) {
        tokens.add(current.toString())
    }
    return tokens
}

private fun buildGreeting(hasFileOps: Boolean, hasTerminalOps: Boolean): String {
    val sections = mutableListOf<String>()
    if (hasFileOps) {
        sections.add(
            "File ops: `<fs.read path=\"/path\" line=\"1\" limit=\"10\">/path</fs.read>` or `<fs.write path=\"/path\">` then content, end with `</fs.write>`."
        )
    }
    if (hasTerminalOps) {
        sections.add(
            "Terminal ops: `<term.create>ls -la</term.create>` or `<term.create>{\"command\":\"ls\",\"args\":[\"-la\"],\"cwd\":\"/tmp\",\"env\":{\"KEY\":\"VALUE\"},\"outputByteLimit\":1024}</term.create>` " +
                "or `<term.output>terminalId</term.output>`, `<term.wait>terminalId</term.wait>`, `<term.kill>terminalId</term.kill>`, `<term.release>terminalId</term.release>`."
        )
    }
    if (sections.isNotEmpty()) {
        sections.add(
            "Tool responses are wrapped as `<tool_response type=\"...\">` ... `</tool_response>`."
        )
    }
    if (sections.isEmpty()) return ""
    return "Hello! Available commands:\n" + sections.joinToString("\n")
}

private data class McpResources(
    val toolCallbacks: List<ToolCallback>,
    private val clients: List<McpSyncClient>
) {
    fun close() {
        clients.forEach { client ->
            runCatching { client.closeGracefully() }
                .onFailure {
//                    logger.warn(it) { "Failed to close MCP client ${client.clientInfo.name()}" }
                }
        }
    }
}

private fun buildMcpResources(servers: List<McpServer>): McpResources {
    if (servers.isEmpty()) return McpResources(emptyList(), emptyList())
    val objectMapper = ObjectMapper()
    val jsonMapper = JacksonMcpJsonMapper(objectMapper)
    val clients = mutableListOf<McpSyncClient>()
    val toolCallbacks = mutableListOf<ToolCallback>()

    servers.forEach { server ->
        val client = when (server) {
            is McpServer.Http -> buildHttpMcpClient(server, jsonMapper)
            is McpServer.Stdio -> buildStdioMcpClient(server, jsonMapper)
            is McpServer.Sse -> {
//                logger.warn { "Skipping unsupported MCP SSE server ${server.name}" }
                null
            }
        }

        if (client != null) {
            clients.add(client)
            val callbacks = buildToolCallbacks(server.name, client, objectMapper)
            toolCallbacks.addAll(callbacks)
        }
    }

    return McpResources(toolCallbacks, clients)
}

private fun buildHttpMcpClient(
    server: McpServer.Http,
    jsonMapper: JacksonMcpJsonMapper
): McpSyncClient? {
    return runCatching {
        val uri = URI.create(server.url)
        val baseUrl = URI(uri.scheme, uri.userInfo, uri.host, uri.port, null, null, null).toString()
        val endpoint = buildHttpEndpoint(uri)
        val builder = HttpClientStreamableHttpTransport.builder(baseUrl)
            .endpoint(endpoint)
            .jsonMapper(jsonMapper)

        if (server.headers.isNotEmpty()) {
            builder.customizeRequest { request ->
                server.headers.forEach { header -> request.header(header.name, header.value) }
            }
        }

        val transport = builder.build()
        val client = McpClient.sync(transport).build()
        client.initialize()
        client
    }.getOrElse { ex ->
//        logger.warn(ex) { "Failed to initialize MCP HTTP client ${server.name} (${server.url})" }
        null
    }
}

private fun buildStdioMcpClient(
    server: McpServer.Stdio,
    jsonMapper: JacksonMcpJsonMapper
): McpSyncClient? {
    return runCatching {
        val env = server.env.associate { it.name to it.value }
        val params = ServerParameters.builder(server.command)
            .args(server.args)
            .env(env)
            .build()
        val transport = StdioClientTransport(params, jsonMapper)
        val client = McpClient.sync(transport).build()
        client.initialize()
        client
    }.getOrElse { ex ->
//        logger.warn(ex) { "Failed to initialize MCP stdio client ${server.name}" }
        null
    }
}

private fun buildToolCallbacks(
    serverName: String,
    client: McpSyncClient,
    objectMapper: ObjectMapper
): List<ToolCallback> {
    val tools = runCatching { client.listTools().tools() }.getOrElse { ex ->
//        logger.warn(ex) { "Failed to list MCP tools for $serverName" }
        emptyList()
    }
    val namePrefix = sanitizeServerName(serverName)
    return tools.map { tool ->
        val toolName = "$namePrefix.${tool.name()}"
        val toolDefinition = DefaultToolDefinition.builder()
            .name(toolName)
            .description(tool.description())
            .inputSchema(objectMapper.writeValueAsString(tool.inputSchema()))
            .build()
        McpToolCallback(toolDefinition, client, tool.name(), objectMapper)
    }
}

private fun sanitizeServerName(name: String): String {
    return name.trim().replace(Regex("\\s+"), "_")
}

private fun buildHttpEndpoint(uri: URI): String {
    val path = uri.rawPath?.takeIf { it.isNotBlank() } ?: "/mcp"
    val query = uri.rawQuery?.let { "?$it" } ?: ""
    return "$path$query"
}

private class McpToolCallback(
    private val toolDefinition: org.springframework.ai.tool.definition.ToolDefinition,
    private val client: McpSyncClient,
    private val toolName: String,
    private val objectMapper: ObjectMapper
) : ToolCallback {
    override fun getToolDefinition(): org.springframework.ai.tool.definition.ToolDefinition = toolDefinition

    override fun call(toolInput: String): String = call(toolInput, null)

    override fun call(
        toolInput: String,
        toolContext: org.springframework.ai.chat.model.ToolContext?
    ): String {
        return runCatching {
            val args = if (toolInput.isBlank()) {
                emptyMap<String, Any>()
            } else {
                objectMapper.readValue(toolInput, object : TypeReference<Map<String, Any>>() {})
            }
            val result = client.callTool(McpSchema.CallToolRequest(toolName, args))
            formatMcpToolResult(result, objectMapper)
        }.getOrElse { ex ->
//            logger.warn(ex) { "MCP tool $toolName failed" }
//            """{ "error": "true", "message": "${ex.message ?: "Tool call failed"}" }"""
            ""
        }
    }
}

private fun formatMcpToolResult(
    result: McpSchema.CallToolResult,
    objectMapper: ObjectMapper
): String {
    if (result.isError() == true && result.content().isNullOrEmpty()) {
        return """{ "error": "true", "message": "Unknown error" }"""
    }
    val content = result.content()
    if (content.isNullOrEmpty()) return ""
    val first = content.first()
    return when (first) {
        is McpSchema.TextContent -> first.text()
        is McpSchema.ResourceLink -> first.uri()
        else -> objectMapper.writeValueAsString(first)
    }
}

public class OpenAiClient(
    private val apiKey: String,
    private val baseUrl: String,
    val model: String,
    val systemPrompt: String,
    private val mockResponse: String?
) {
    val h = CollectionUtils.toMultiValueMap(mutableMapOf(Pair("Authorization", mutableListOf("Bearer " + apiKey))))

    private val openAiApi = OpenAiApi.builder()
        .apiKey(apiKey)
        .baseUrl(baseUrl)
        .headers(h)
        .build()

    private val chatModel = OpenAiChatModel.builder()
        .openAiApi(openAiApi)
        .defaultOptions(
            OpenAiChatOptions.builder()
                .model(model)
                .temperature(0.2)
                .build()
        )
        .build()

    fun createChatCompletion(
        messages: List<Message>,
        toolCallbacks: List<ToolCallback>,
        toolCallingManager: DefaultToolCallingManager
    ): ChatCompletionResult {
        mockResponse?.let {
            val assistant = AssistantMessage(it)
            return ChatCompletionResult(it, messages + assistant)
        }

        val options = OpenAiChatOptions.builder()
            .model(model)
            .temperature(0.2)
            .toolCallbacks(toolCallbacks)
            .internalToolExecutionEnabled(false)
            .build()

        var prompt = Prompt(messages, options)
        var response = chatModel.call(prompt)

        if (response.hasToolCalls()) {
            val toolResult = toolCallingManager.executeToolCalls(prompt, response)
            val toolHistory = toolResult.conversationHistory()
            if (toolResult.returnDirect()) {
                val generation = ToolExecutionResult.buildGenerations(toolResult).firstOrNull()
                val assistant = generation?.output
                val content = assistant?.text.orEmpty()
                val history = if (assistant != null) toolHistory + assistant else toolHistory
                return ChatCompletionResult(content, history)
            }

            prompt = Prompt(toolHistory, options)
            response = chatModel.call(prompt)
            val assistant = response.result?.output
            val content = assistant?.text.orEmpty()
            val history = if (assistant != null) toolHistory + assistant else toolHistory
            return ChatCompletionResult(content, history)
        }

        val assistant = response.result?.output
        val content = assistant?.text.orEmpty()
        val history = if (assistant != null) messages + assistant else messages
        return ChatCompletionResult(content, history)
    }

    companion object {
        fun fromEnv(): OpenAiClient {
            val apiKey = envOrThrow("OPENAI_API_KEY")
            val baseUrl = env("OPENAI_BASE_URL") ?: "https://api.openai.com/v1"
            val model = env("OPENAI_MODEL") ?: "gpt-4o-mini"
            val systemPrompt = env("OPENAI_SYSTEM_PROMPT") ?: "You are a helpful ACP agent."
            val mockResponse = env("OPENAI_MOCK_RESPONSE")
            return OpenAiClient(apiKey, baseUrl, model, systemPrompt, mockResponse)
        }

        private fun env(name: String): String? = System.getenv(name)?.takeIf { it.isNotBlank() }

        private fun envOrThrow(name: String): String {
            return env(name) ?: error("Missing required environment variable: $name")
        }
    }
}

data class ChatCompletionResult(
    val content: String,
    val updatedHistory: List<Message>
)
