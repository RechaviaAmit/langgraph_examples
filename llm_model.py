from langchain_openai import AzureChatOpenAI


model_config = {
    'openai_api_type': 'azure',
    'openai_api_key': '',
    'deployment_name': 'gpt-4-turbo-2024-04-09',
    'openai_api_version': '2024-07-01-preview',
    'azure_endpoint': ''
}


model = AzureChatOpenAI(
    **model_config
)