rm -rf ../../test-acp-client/build
./gradlew samples:kotlin-acp-openai-agent:build -x test 
./gradlew samples:kotlin-acp-openai-agent:install 
cp -r samples/kotlin-acp-openai-agent/build ../../test-acp-client/build
