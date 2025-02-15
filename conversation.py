import dataclasses
from enum import auto, Enum
from typing import List

class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()


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

        elif self.sep_style == SeparatorStyle.LLAMA_2:
            # wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            # wrap_inst = lambda msg: f"[INST] {msg} [/INST]"

            def wrap_inst(msg):
                return f"[INST] {msg} [/INST]"
            
            def wrap_sys(msg):
                return f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg

            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    
                    if i == 0: 
                        message = wrap_sys(self.system) + message
                    
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            
            def wrap_inst(role, msg):
                return f"<|start_header_id|>{role}<|end_header_id|>{msg}{self.sep}"
                
            ret = "<|begin_of_text|>"

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"

                    ret += wrap_inst("system", self.system)

                ret += wrap_inst(role, message)

            return ret

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

llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

llama_3 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("user", "assistant"),
    version="llama_v3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
    sep2=None,
)

default_conversation = llama_3