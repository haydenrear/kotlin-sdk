package com.agentclientprotocol.samples.openai

import com.agentclientprotocol.agent.Agent
import com.agentclientprotocol.client.Client
import com.agentclientprotocol.client.ClientInfo
import com.agentclientprotocol.common.Event
import com.agentclientprotocol.common.FileSystemOperations
import com.agentclientprotocol.common.SessionCreationParameters
import com.agentclientprotocol.common.TerminalOperations
import com.agentclientprotocol.model.ClientCapabilities
import com.agentclientprotocol.model.ContentBlock
import com.agentclientprotocol.model.EnvVariable
import com.agentclientprotocol.model.FileSystemCapability
import com.agentclientprotocol.model.HttpHeader
import com.agentclientprotocol.model.McpServer
import com.agentclientprotocol.model.SessionUpdate
import com.agentclientprotocol.model.ToolCallContent
import com.agentclientprotocol.model.ToolCallStatus
import com.agentclientprotocol.protocol.Protocol
import com.agentclientprotocol.transport.StdioTransport
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.toList
import kotlinx.io.asSink
import kotlinx.io.asSource
import kotlinx.io.buffered
import java.nio.channels.Channels
import java.nio.channels.Pipe
import java.nio.file.Files
import java.nio.file.Path
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class OpenAiAcpAgentOperationsTest {
    @Test
    fun testFileAndTerminalOperations() = runBlocking {
        val tempDir = Files.createTempDirectory("acp-openai-agent-test")
        val testFile = tempDir.resolve("acp_test.txt")
        val testContent = "hello from acp openai agent"

        val clientToAgent = Pipe.open()
        val agentToClient = Pipe.open()

        val clientTransport = StdioTransport(
            this,
            Dispatchers.IO,
            input = Channels.newInputStream(agentToClient.source()).asSource().buffered(),
            output = Channels.newOutputStream(clientToAgent.sink()).asSink().buffered(),
            "client"
        )

        val agentTransport = StdioTransport(
            this,
            Dispatchers.IO,
            input = Channels.newInputStream(clientToAgent.source()).asSource().buffered(),
            output = Channels.newOutputStream(agentToClient.sink()).asSink().buffered(),
            "agent"
        )

        val clientProtocol = Protocol(this, clientTransport)
        val agentProtocol = Protocol(this, agentTransport)
        clientProtocol.start()
        agentProtocol.start()

        val openAiClient = OpenAiClient(
            apiKey = "local",
            baseUrl = "http://localhost:11434",
            model = "qwen3-coder:30b",
            systemPrompt = """
                You are a helpful ACP agent.
                When asked to perform file or terminal operations, respond only with ACP directives in XML.
                Use these formats (start tags must be at the beginning of a line):
                - <fs.write path="/path"> then content lines, end with </fs.write>
                - <fs.read path="/path" line="1" limit="10">/path</fs.read>
                - <term.create>command [args...]</term.create> or <term.create>{"command":"ls","args":["-la"],"cwd":"/tmp","env":{"KEY":"VALUE"},"outputByteLimit":1024}</term.create>
                - <term.wait>terminalId</term.wait>
                - <term.output>terminalId</term.output>
                - <term.release>terminalId</term.release>
                - <term.kill>terminalId</term.kill>
                Do not include extra commentary. Please use fs.read to read, fs.write to write instead of using
                terminal commands.
            """.trimIndent(),
            mockResponse = null
        )

        Agent(
            protocol = agentProtocol,
            agentSupport = OpenAiAgentSupport(openAiClient)
        )

        val client = Client(clientProtocol)
        client.initialize(
            ClientInfo(
                capabilities = ClientCapabilities(
                    fs = FileSystemCapability(readTextFile = true, writeTextFile = true),
                    terminal = true
                )
            )
        )

        val session = client.newSession(
            SessionCreationParameters(tempDir.toString(), emptyList())
        ) { _, _ -> TestClientOperations(tempDir) }

        val writeInstruction = "Write a file at ${testFile.toString()} with the exact content: $testContent"
        val writeCommand = promptForCommand(session, writeInstruction, listOf("<fs.write"))
        assertMatchesXmlCommand(writeCommand, "fs.write", testFile.toString())
        assertTrue(writeCommand.contains("</fs.write>"), writeCommand)

        val writeEvents = session.prompt(listOf(ContentBlock.Text(writeCommand))).toList()
        val writeToolMessages = extractToolMessages(writeEvents)
        assertEquals(testContent, testFile.readText())
        assertTrue(writeToolMessages.any { it.contains("Wrote ${testFile.toString()}") })

        val readInstruction = "Read the file at ${testFile.toString()} using your read command and return the correct ACP command."
        val readCommand = promptForCommand(session, readInstruction, listOf("<fs.read"))
        assertMatchesXmlCommand(readCommand, "fs.read", testFile.toString())

        val readEvents = session.prompt(listOf(ContentBlock.Text(readCommand))).toList()
        val readToolMessages = extractToolMessages(readEvents)
        assertTrue(readToolMessages.any { it.contains("Read ${testFile.toString()}") && it.contains(testContent) })

        val terminalInstruction = "Run a terminal command that outputs the file at ${testFile.toString()} using <term.create>."
        val terminalCreateCommand = promptForCommand(session, terminalInstruction, listOf("<term.create"))
        assertTrue(terminalCreateCommand.contains("<term.create"), terminalCreateCommand)

        val terminalCreateEvents = session.prompt(listOf(ContentBlock.Text(terminalCreateCommand))).toList()
        val terminalCreateMessages = extractToolMessages(terminalCreateEvents)
        val terminalId = terminalCreateMessages
            .firstOrNull { it.startsWith("Created terminal:") }
            ?.substringAfter("Created terminal:")
            ?.trim()
        assertNotNull(terminalId, "Terminal id should be returned from /term.create")

        val terminalFollowupInstruction = "Provide ACP commands to wait for, output, and release terminal $terminalId, respectively, using the XML tags."
        val terminalFollowupCommands = promptForCommand(session, terminalFollowupInstruction, listOf("<term.wait", "<term.output", "<term.release"))
        assertMatchesXmlCommand(terminalFollowupCommands, "term.wait", terminalId)
        assertMatchesXmlCommand(terminalFollowupCommands, "term.output", terminalId)
        assertMatchesXmlCommand(terminalFollowupCommands, "term.release", terminalId)

        val terminalFollowupEvents = session.prompt(listOf(ContentBlock.Text(terminalFollowupCommands))).toList()
        val terminalFollowupMessages = extractToolMessages(terminalFollowupEvents)
        assertTrue(terminalFollowupMessages.any { it.contains("Terminal exit ($terminalId)") })
        assertTrue(terminalFollowupMessages.any { it.contains("Terminal output ($terminalId)") && it.contains(testContent) })

        agentProtocol.close()
        clientProtocol.close()
    }

    @Test
    fun testMcpHttpServerToolsAvailable() = runBlocking {
        val clientToAgent = Pipe.open()
        val agentToClient = Pipe.open()

        val clientTransport = StdioTransport(
            this,
            Dispatchers.IO,
            input = Channels.newInputStream(agentToClient.source()).asSource().buffered(),
            output = Channels.newOutputStream(clientToAgent.sink()).asSink().buffered(),
            "client"
        )

        val agentTransport = StdioTransport(
            this,
            Dispatchers.IO,
            input = Channels.newInputStream(clientToAgent.source()).asSource().buffered(),
            output = Channels.newOutputStream(agentToClient.sink()).asSink().buffered(),
            "agent"
        )

        val clientProtocol = Protocol(this, clientTransport)
        val agentProtocol = Protocol(this, agentTransport)
        clientProtocol.start()
        agentProtocol.start()

//        val server = McpServer.Http(
//            name = "deepwiki",
//            url = "https://mcp.deepwiki.com/mcp",
//            headers = emptyList()
//        )

        val servers = mutableListOf<McpServer>()

        val openAiClient = OpenAiClient(
            apiKey = "local",
            baseUrl = "http://localhost:11434",
            model = "qwen3-coder:30b",
            systemPrompt = """
                You are a helpful ACP agent.
                When asked about MCP tools, respond with a comma-separated list of tool names exactly as available.
                Include at least one tool name and do not add extra commentary.
            """.trimIndent(),
            mockResponse = null
        )


        val agentSupport = OpenAiAgentSupport(openAiClient)
        Agent(
            protocol = agentProtocol,
            agentSupport = agentSupport
        )

        val client = Client(clientProtocol)
        client.initialize(
            ClientInfo(
                capabilities = ClientCapabilities()
            )
        )

        val session = client.newSession(
            SessionCreationParameters(System.getProperty("user.dir"), servers)
        ) { _, _ -> TestClientOperations(Path.of(System.getProperty("user.dir"))) }

        val toolPrompt = "List the MCP tool names you can access from deepwiki."
        val response = promptForCommand(session, toolPrompt, listOf("deepwiki."))

        agentProtocol.close()
        clientProtocol.close()
    }

    private fun extractToolMessages(events: List<Event>): List<String> {
        return events.filterIsInstance<Event.SessionUpdateEvent>()
            .map { it.update }
            .filterIsInstance<SessionUpdate.ToolCallUpdate>()
            .filter { it.status == ToolCallStatus.COMPLETED }
            .flatMap { update ->
                update.content.orEmpty().mapNotNull { content ->
                    val block = (content as? ToolCallContent.Content)?.content
                    (block as? ContentBlock.Text)?.text?.let { unwrapToolResponse(it) }
                }
            }
    }

    private suspend fun promptForCommand(
        session: com.agentclientprotocol.client.ClientSession,
        instruction: String,
        expectedTokens: List<String>
    ): String {
        val events = session.prompt(listOf(ContentBlock.Text(instruction))).toList()
        val chunks = extractAgentTextChunks(events)
        val candidate = chunks.firstOrNull { chunk ->
            !chunk.startsWith("Hello! Available commands:") &&
                expectedTokens.any { token -> chunk.contains(token) }
        } ?: chunks.lastOrNull().orEmpty()
        assertTrue(candidate.isNotBlank(), "Expected agent response for instruction: $instruction. Chunks=$chunks")
        return stripCodeFence(candidate.trim())
    }

    private fun extractAgentTextChunks(events: List<Event>): List<String> {
        return events.filterIsInstance<Event.SessionUpdateEvent>()
            .map { it.update }
            .filterIsInstance<SessionUpdate.AgentMessageChunk>()
            .mapNotNull { update ->
                (update.content as? ContentBlock.Text)?.text
            }
    }

    private fun stripCodeFence(text: String): String {
        val trimmed = text.trim()
        if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
            return trimmed.removePrefix("```").removeSuffix("```").trim()
        }
        return trimmed
    }

    private fun assertMatchesXmlCommand(commandBlock: String, tag: String, argument: String) {
        val escapedArgument = Regex.escape(argument)
        val pattern = when (tag) {
            "fs.write" -> Regex("<${Regex.escape(tag)}\\b[^>]*\\bpath\\s*=\\s*\"$escapedArgument\"[^>]*>")
            "fs.read" -> Regex(
                "<${Regex.escape(tag)}\\b[^>]*\\bpath\\s*=\\s*\"$escapedArgument\"[^>]*>" +
                    "|<${Regex.escape(tag)}>\\s*$escapedArgument\\s*</${Regex.escape(tag)}>"
            )
            else -> Regex("<${Regex.escape(tag)}>\\s*$escapedArgument\\s*</${Regex.escape(tag)}>")
        }
        assertTrue(pattern.containsMatchIn(commandBlock), commandBlock)
    }

    private fun unwrapToolResponse(text: String): String {
        val trimmed = text.trim()
        val openTag = Regex("^<tool_response\\b[^>]*>")
        if (!openTag.containsMatchIn(trimmed)) return trimmed
        val withoutOpen = trimmed.replaceFirst(openTag, "")
        val closeIndex = withoutOpen.lastIndexOf("</tool_response>")
        return if (closeIndex >= 0) {
            withoutOpen.substring(0, closeIndex).trim()
        } else {
            withoutOpen.trim()
        }
    }
}

private class TestClientOperations(private val cwd: Path) : com.agentclientprotocol.common.ClientSessionOperations, FileSystemOperations, TerminalOperations {
    private val activeTerminals = ConcurrentHashMap<String, Process>()

    override suspend fun requestPermissions(
        toolCall: SessionUpdate.ToolCallUpdate,
        permissions: List<com.agentclientprotocol.model.PermissionOption>,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.RequestPermissionResponse {
        return com.agentclientprotocol.model.RequestPermissionResponse(
            com.agentclientprotocol.model.RequestPermissionOutcome.Selected(permissions.first().optionId)
        )
    }

    override suspend fun notify(
        notification: SessionUpdate,
        _meta: kotlinx.serialization.json.JsonElement?
    ) {
        // no-op for test
    }

    override suspend fun fsReadTextFile(
        path: String,
        line: UInt?,
        limit: UInt?,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.ReadTextFileResponse {
        val content = Path.of(path).readText()
        return com.agentclientprotocol.model.ReadTextFileResponse(content)
    }

    override suspend fun fsWriteTextFile(
        path: String,
        content: String,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.WriteTextFileResponse {
        Path.of(path).writeText(content)
        return com.agentclientprotocol.model.WriteTextFileResponse()
    }

    override suspend fun terminalCreate(
        command: String,
        args: List<String>,
        cwd: String?,
        env: List<EnvVariable>,
        outputByteLimit: ULong?,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.CreateTerminalResponse {
        val processBuilder = ProcessBuilder(listOf(command) + args)
        processBuilder.directory(java.io.File(cwd ?: this.cwd.toString()))
        env.forEach { processBuilder.environment()[it.name] = it.value }

        val process = processBuilder.start()
        val terminalId = UUID.randomUUID().toString()
        activeTerminals[terminalId] = process
        return com.agentclientprotocol.model.CreateTerminalResponse(terminalId)
    }

    override suspend fun terminalOutput(
        terminalId: String,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.TerminalOutputResponse {
        val process = activeTerminals[terminalId] ?: error("Terminal not found: $terminalId")
        val stdout = process.inputStream.bufferedReader().readText()
        val stderr = process.errorStream.bufferedReader().readText()
        val output = if (stderr.isNotEmpty()) "$stdout\nSTDERR:\n$stderr" else stdout
        return com.agentclientprotocol.model.TerminalOutputResponse(output, truncated = false)
    }

    override suspend fun terminalRelease(
        terminalId: String,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.ReleaseTerminalResponse {
        activeTerminals.remove(terminalId)
        return com.agentclientprotocol.model.ReleaseTerminalResponse()
    }

    override suspend fun terminalWaitForExit(
        terminalId: String,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.WaitForTerminalExitResponse {
        val process = activeTerminals[terminalId] ?: error("Terminal not found: $terminalId")
        val exitCode = process.waitFor()
        return com.agentclientprotocol.model.WaitForTerminalExitResponse(exitCode.toUInt())
    }

    override suspend fun terminalKill(
        terminalId: String,
        _meta: kotlinx.serialization.json.JsonElement?
    ): com.agentclientprotocol.model.KillTerminalCommandResponse {
        val process = activeTerminals[terminalId]
        process?.destroy()
        return com.agentclientprotocol.model.KillTerminalCommandResponse()
    }
}
