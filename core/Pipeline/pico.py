from Configs import config
from Agents import agent

PROJECT_ID = config.PROJECT_ID
LOCATION = config.LOCATION
ENDPOINT_ID = config.ENDPOINT_ID
 
# 1) create an agent by model name (via registry)
# MODEL_REGISTRY["gemma-3-4b-it"] = "mg-endpoint-XXXX..."  # set this once
agent = agent.agent_factory(PROJECT_ID, LOCATION, "gemma-3-4b-it", temperature=0.2, endpoint_id=ENDPOINT_ID)

# 2) system + chat
agent.set_system("You are concise and helpful.")
print(agent.say("write down what is quantum mechanics in detail"))

# 3) single-shot completion (ignores history)
print(agent.complete("Give me at least 10 5-word slogan for healthy snacks."))

# 5) reset
agent.reset()
