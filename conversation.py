import dataclasses
from enum import auto, Enum
from typing import List

class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False


    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version
        )
    

    def append_message(self, role, message):
        self.messages.append([role, message])

    
    def get_prompt(self):
        messages = self.messages

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        raise ValueError(f"Tuple message {message} found")
                    
                    ret += role + ": " + message + seps[i % 2]

                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
        
        return ret


jamba = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("Human", "Assistant"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
    sep2=None,
) 

default_conversation = jamba