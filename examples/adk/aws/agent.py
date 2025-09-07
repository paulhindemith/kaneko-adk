"""
Local data agent
"""
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field


class OutputSchema(BaseModel):
    candidate1: str = Field(..., description="回答の候補1")
    candidate2: str = Field(..., description="回答の候補2")
    candidate3: str = Field(..., description="回答の候補3")


root_agent = LlmAgent(
    name="aws",
    instruction="日本語で答えてください。",
    model=LiteLlm(
        model="bedrock/converse/apac.anthropic.claude-sonnet-4-20250514-v1:0"
    ),
    output_schema=OutputSchema,
)
