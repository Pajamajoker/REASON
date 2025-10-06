from Configs import config
from Agents import agent

PROJECT_ID = config.PROJECT_ID
LOCATION = config.LOCATION
ENDPOINT_ID = config.ENDPOINT_ID
 
# 1) create an agent by model name (via registry)
# MODEL_REGISTRY["gemma-3-4b-it"] = "mg-endpoint-XXXX..."  # set this once
agent = agent.agent_factory(project_id=PROJECT_ID, 
                            location=LOCATION, 
                            temperature=0.2, 
                            endpoint_id=ENDPOINT_ID,
                            dedicated_dns_or_predict_url="586173838023196672.us-east1-409780034045.prediction.vertexai.goog",
                            model="google/gemma3@gemma-3-1b-it",
                            max_tokens=2096)

# 2) system + chat
agent.set_system("You are concise and helpful.")
print(agent.say("write down what is quantum mechanics in detail"))

# 3) single-shot completion (ignores history)
print(agent.complete("Give me at least 10 5-word slogan for healthy snacks."))

# 5) reset
agent.reset()
