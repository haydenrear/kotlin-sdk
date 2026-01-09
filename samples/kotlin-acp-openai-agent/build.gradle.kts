plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
    application
}

dependencies {
    implementation(project(":acp"))
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlin.logging)
    implementation(libs.kotlinx.serialization.json)
    implementation("org.springframework.ai:spring-ai-starter-model-openai:1.1.2")
    implementation("org.springframework.ai:spring-ai-starter-mcp-client:1.1.2")
    implementation("org.springframework.ai:spring-ai-mcp:1.1.2")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.17.2")
    implementation("ch.qos.logback:logback-classic:1.5.13")
    testImplementation(kotlin("test"))
}

application {
    mainClass.set("com.agentclientprotocol.samples.openai.OpenAiAcpAgentAppKt")
}

kotlin {
    jvmToolchain(21)
}
