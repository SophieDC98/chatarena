import os
import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME, Message
from .base import IntelligenceBackend, register_backend

try:
    import anthropic
except ImportError:
    is_anthropic_available = False
    # logging.warning("anthropic package is not installed")
else:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key is None:
        # logging.warning("Anthropic API key is not set. Please set the environment variable ANTHROPIC_API_KEY")
        is_anthropic_available = False
    else:
        is_anthropic_available = True

DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."

@register_backend
class Claude(IntelligenceBackend):
    """Interface to the Claude offered by Anthropic."""

    stateful = False
    type_name = "claude"

    def __init__(
        self, 
        max_tokens: int = DEFAULT_MAX_TOKENS, 
        model: str = DEFAULT_MODEL,
        merge_other_agents_as_one_user: bool = True, 
        **kwargs
    ):
        assert (
            is_anthropic_available
        ), "anthropic package is not installed or the API key is not set"
        super().__init__(max_tokens=max_tokens, model=model, merge_other_agents_as_one_user=merge_other_agents_as_one_user, **kwargs)

        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages, system):
        response = self.client.messages.create(
            messages=messages,
            system=system,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model,
            max_tokens=self.max_tokens,
        )

        response = response.content[0].text.strip()
        return response

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Format the input and call the Claude API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """
        # all_messages = (
        #     [(SYSTEM_NAME, global_prompt), (SYSTEM_NAME, role_desc)]
        #     if global_prompt
        #     else [(SYSTEM_NAME, role_desc)]
        # )

        # for message in history_messages:
        #     all_messages.append((message.agent_name, message.content))
        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"You are a helpful assistant.\n{global_prompt.strip()}\n{BASE_PROMPT}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            system_prompt = f"You are a helpful assistant. Your name is {agent_name}.\n\nYour role:{role_desc}\n\n{BASE_PROMPT}"

        all_messages = [(SYSTEM_NAME, system_prompt)]
        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                all_messages.append((SYSTEM_NAME, msg.content))
            else:  # non-system messages are suffixed with the end of message token
                all_messages.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))
        if request_msg:
            all_messages.append((SYSTEM_NAME, request_msg.content))

        messages = []
        for i, msg in enumerate(all_messages):
            if i == 0:
                assert (
                    msg[0] == SYSTEM_NAME
                )  # The first message should be from the system
                messages.append({"role": "user", "content": msg[1]})
                system =  msg[1]
            else:
                if msg[0] == agent_name:
                    messages.append({"role": "assistant", "content": msg[1]})
                else:
                    if messages[-1]["role"] == "user":  # last message is from user
                        if self.merge_other_agent_as_user:
                            messages[-1][
                                "content"
                            ] = f"{messages[-1]['content']}\n\n[{msg[0]}]: {msg[1]}"
                        else:
                            messages.append(
                                {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                            )
                    elif (
                        (messages[-1]["role"] == "assistant") or (messages[-1]["role"] == "system")
                    ):  # consecutive assistant messages
                        # Merge the assistant messages
                        messages.append(
                            {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                        )
                    else:
                        raise ValueError(f"Invalid role: {messages[-1]['role']}")

        response = self._get_response(messages, system, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()  # noqa: F541
        response = re.sub(
            rf"^\s*{re.escape(agent_name)}\s*:", "", response
        ).strip()  # noqa: F451

        # Remove the tailing end of message token
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        return response
