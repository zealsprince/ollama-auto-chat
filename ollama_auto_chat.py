import json
import sys
import threading
import time
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from rich import print
from rich.console import Console
from rich.prompt import Prompt, Confirm

# Configuration
OLLAMA_HOST = "http://localhost:11434"

# Initialize console
console = Console()

# Define paths for saving conversations
BASE_DIR = Path(__file__).parent
CONVO_DIR = BASE_DIR / "convo"


class ConversationManager:
    def __init__(self):
        self.conversation: List[Dict[str, Any]] = []
        self.ensure_convo_dir()

    def ensure_convo_dir(self) -> None:
        """
        Ensures that the convo directory exists.
        """

        CONVO_DIR.mkdir(exist_ok=True)

        console.print(
            f"[green]Conversation directory ensured at {CONVO_DIR}[/green]"
        )

    def list_existing_conversations(self) -> List[Path]:
        """
        Lists existing conversation folders.
        """

        return [item for item in CONVO_DIR.iterdir() if item.is_dir()]

    def create_new_conversation_folder(self) -> Path:
        """
        Creates a new conversation folder with a timestamp.
        """

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        convo_folder = CONVO_DIR / f"{timestamp}_conversation"
        convo_folder.mkdir()

        console.print(
            f"[green]Created new conversation folder at {convo_folder}[/green]"
        )

        return convo_folder

    def save_conversation(
        self, convo_folder: Path, silent: bool = False
    ) -> None:
        """
        Saves the conversation history to a history.json file.
        """

        history_file = convo_folder / "history.json"
        try:
            with history_file.open('w', encoding='utf-8') as f:
                json.dump(
                    self.conversation,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            if not silent:
                console.print(
                    f"[green]Conversation saved to {history_file}[/green]"
                )

        except Exception as e:
            console.print(
                f"[red]Error saving conversation: {e}[/red]"
            )

    def load_conversation(self, convo_folder: Path) -> None:
        """
        Loads the conversation history from a history.json file.
        """

        history_file = convo_folder / "history.json"
        if not history_file.exists():
            console.print(
                f"[red]No conversation history found at {history_file}[/red]"
            )

            return

        try:
            with history_file.open('r', encoding='utf-8') as f:
                loaded_conversation = json.load(f)
                self.conversation.extend(loaded_conversation)

            console.print(
                f"[green]Loaded conversation from {history_file}[/green]"
            )

        except json.JSONDecodeError:
            console.print(
                f"[red]Error decoding JSON from {history_file}[/red]"
            )

    def append_message(self, role: str, content: str) -> None:
        """
        Appends a message to the conversation history.
        """

        self.conversation.append({"role": role, "content": content})

    def display_conversation(
        self, roles: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Displays the loaded conversation history for continuity.
        """

        console.print(
            "\n[bold underline]Previous Conversation:[/bold underline]\n"
        )

        for message in self.conversation:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":
                console.print(
                    f"[bold blue]{roles['user']['name']}[/bold blue]: {content}\n"
                )

            elif role == "assistant":
                console.print(
                    f"[bold green]{roles['assistant']['name']}[/bold green]: {content}\n"
                )

            elif role == "system":
                console.print(
                    f"[bold magenta]System:[/bold magenta] {content}\n"
                )

        # Add a separator after displaying the conversation
        console.print("\n" + "-" * 50 + "\n")


class OllamaAPI:
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host

    def list_local_models(self) -> List[Dict[str, Any]]:
        """
        Fetches the list of local models from the Ollama API.
        """

        url = f"{self.host}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])

            return models

        except requests.exceptions.RequestException as e:
            console.print(
                f"[red]Error fetching models: {e}[/red]"
            )

            sys.exit(1)

    def select_model(self, models: List[Dict[str, Any]]) -> str:
        """
        Prompts the user to select a model from the available list.
        """

        if not models:
            console.print(
                "[red]No models available. Please load or create a model first.[/red]"
            )

            sys.exit(1)

        console.print("\nAvailable Models:")

        for idx, model in enumerate(models, start=1):
            name = model.get("name", "Unnamed Model")
            size = model.get("size", "Unknown Size")
            details = model.get("details", {})

            param_size = details.get(
                "parameter_size", "Unknown Parameters"
            )
            quant_level = details.get(
                "quantization_level", "Unknown Quantization"
            )

            console.print(
                f"[cyan]{idx}.[/cyan] [bold]{name}[/bold] - Parameters: {param_size}, "
                f"Quantization: {quant_level}, Size: {size} bytes"
            )

        while True:
            choice = Prompt.ask(
                "\nSelect a model by number (or type 'exit' to quit)",
                default="1"
            )

            if choice.lower() in ['exit', 'quit']:
                console.print("Exiting application. Goodbye!")
                sys.exit(0)

            if not choice.isdigit():
                console.print(
                    "[red]Please enter a valid number.[/red]"
                )

                continue

            choice = int(choice)
            if 1 <= choice <= len(models):
                selected_model = models[choice - 1]['name']
                console.print(
                    f"[green]Selected model: {selected_model}[/green]"
                )

                return selected_model

            else:
                console.print(
                    "[red]Choice out of range. Please select a valid number.[/red]"
                )

    def send_chat_request_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[Any] = None
    ):
        """
        Sends a chat completion request to the Ollama API and yields parts of the assistant's response as they arrive.
        """

        url = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": model,
            "messages": messages,
            "stream": True  # Enable streaming
        }

        if options:
            payload["options"] = options

        if tools:
            payload["tools"] = tools

        try:
            with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            json_line = json.loads(decoded_line)
                            if ('message' in json_line and 'content' in json_line['message']):
                                yield json_line['message']['content']

                            elif 'done' in json_line and json_line['done']:
                                break

                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            console.print(
                f"[red]Error communicating with Ollama API: {e}[/red]"
            )

            return


def get_role_descriptions() -> Dict[str, Dict[str, str]]:
    """
    Prompts the user to enter custom names and system prompts for user and assistant roles.
    """

    console.print("\n[bold]Define Roles[/bold]")

    # User Role
    user_name = Prompt.ask(
        "Enter the name for the user role",
        default="User"
    )
    user_description = Prompt.ask(
        "Enter the system prompt for the user",
        default="You are a user engaging in a conversation."
    )

    # Assistant Role
    assistant_name = Prompt.ask(
        "Enter the name for the assistant role",
        default="Assistant"
    )
    assistant_description = Prompt.ask(
        "Enter the system prompt for the assistant",
        default="You are an AI assistant that responds helpfully and accurately."
    )

    return {
        "user": {
            "name": user_name,
            "description": user_description
        },
        "assistant": {
            "name": assistant_name,
            "description": assistant_description
        }
    }


def display_menu(conversations: List[Path]) -> Optional[Path]:
    """
    Prompts the user to select an existing conversation or start a new one.
    """
    if not conversations:
        return None

    console.print("\n[bold yellow]Existing Conversations:[/bold yellow]")

    for idx, convo_folder in enumerate(conversations, start=1):
        name = convo_folder.name
        timestamp = name.split("_")[0]

        console.print(
            f"[cyan]{idx}.[/cyan] [bold]{name}[/bold] - Started at: {timestamp}"
        )

    console.print(
        f"[cyan]{len(conversations) + 1}.[/cyan] Start a New Conversation"
    )

    while True:
        choice = Prompt.ask(
            "\nSelect a conversation to continue or start new",
            default=str(len(conversations) + 1)
        )

        if not choice.isdigit():
            console.print(
                "[red]Please enter a valid number.[/red]"
            )

            continue

        choice = int(choice)
        if 1 <= choice <= len(conversations):
            selected_convo = conversations[choice - 1]
            console.print(
                f"[green]Continuing conversation: {selected_convo.name}[/green]"
            )

            return selected_convo

        elif choice == len(conversations) + 1:
            return None

        else:
            console.print(
                "[red]Choice out of range. Please select a valid number.[/red]"
            )


def select_mode() -> str:
    """
    Prompts the user to select a mode of operation.
    """

    console.print("\n[bold]Select Mode:[/bold]")
    console.print("1. Interactive Chat Mode")
    console.print("2. Automated Conversation Mode")

    while True:
        mode_choice = Prompt.ask(
            "\nEnter the number corresponding to your choice",
            default="1"
        )

        if mode_choice in ['1', '2']:
            return mode_choice

        console.print(
            "[red]Please enter a valid choice (1 or 2).[/red]"
        )


def handle_streaming_response(
    generator: str
) -> str:
    """
    Processes the streaming response from the generator and ensures it matches the expected role.
    """

    full_response = ""
    for chunk in generator:
        full_response += chunk
        console.print(chunk, end='')

    return full_response.strip()


def get_user_input(
    api: OllamaAPI,
    conv_manager: ConversationManager,
    model: str,
    roles: Dict[str, Dict[str, str]],
    options: Optional[Dict[str, Any]],
    convo_folder: Path,
    stop_event: threading.Event
) -> None:
    """
    Continuously listens for user input in a separate thread.
    Handles streaming responses for assistant replies.
    """

    user_name = roles["user"]["name"]
    assistant_name = roles["assistant"]["name"]

    while not stop_event.is_set():
        try:
            user_message = Prompt.ask(
                f"\n[bold blue]{user_name}[/bold blue]"
            )

            if user_message.lower() in ['exit', 'quit']:
                stop_event.set()
                console.print("Exiting chat. Goodbye!")

                conv_manager.save_conversation(
                    convo_folder,
                    silent=True
                )

                sys.exit(0)

            conv_manager.append_message("user", user_message)
            conv_manager.save_conversation(
                convo_folder,
                silent=True
            )

            # Send user message and get assistant response with streaming
            generator = api.send_chat_request_stream(
                model,
                conv_manager.conversation,
                options=options
            )

            console.print(
                f"\n[bold green]{assistant_name}[/bold green]: ",
                end=''
            )

            assistant_reply = handle_streaming_response(generator)

            if assistant_reply:
                console.print("\n")
                conv_manager.append_message(
                    "assistant",
                    assistant_reply
                )

                conv_manager.save_conversation(
                    convo_folder,
                    silent=True
                )

        except KeyboardInterrupt:
            stop_event.set()
            console.print(
                "\n[red]Exiting chat. Goodbye![/red]"
            )

            conv_manager.save_conversation(convo_folder)

            break

        except EOFError:
            stop_event.set()
            console.print(
                "\n[red]Exiting chat. Goodbye![/red]"
            )

            conv_manager.save_conversation(convo_folder)

            break


def generate_impersonated_message_stream(
    api: OllamaAPI,
    conv_manager: ConversationManager,
    model: str,
    roles: Dict[str, Dict[str, str]],
    target_role: str,
    options: Optional[Dict[str, Any]]
) -> Optional[str]:
    """
    Generates a message for the specified target_role (user or assistant) based on the conversation.
    Handles streaming response.
    """

    if target_role.lower() == 'user':
        prompt_content = (
            f"Continue the conversation by generating a message for the "
            f"{roles['user']['name']} based on the assistant's last response."
        )

        role = "user"

    elif target_role.lower() == 'assistant':
        prompt_content = (
            f"Continue the conversation by generating a message for the "
            f"{roles['assistant']['name']} based on the user's last response."
        )

        role = "assistant"

    else:
        console.print(
            "[red]Invalid target role specified for impersonation.[/red]"
        )

        return None

    prompt_message = {"role": role, "content": prompt_content}
    updated_conversation = conv_manager.conversation + [prompt_message]

    generator = api.send_chat_request_stream(
        model,
        updated_conversation,
        options=options
    )

    response_content = handle_streaming_response(generator)

    if response_content:
        console.print("\n")
        conv_manager.append_message(role, response_content)

    return response_content


def auto_conversation(
    api: OllamaAPI,
    conv_manager: ConversationManager,
    model: str,
    roles: Dict[str, Dict[str, str]],
    options: Optional[Dict[str, Any]],
    delay: int = 2
) -> None:
    """
    Handles automated conversation between user and assistant using streaming responses.
    Ensures proper role initiation and alternation.
    """

    user_name = roles["user"]["name"]

    if not conv_manager.conversation:
        # Generate initial user message
        initial_user_prompt = (
            f"Start the conversation as {user_name} about any topic you prefer."
        )

        conv_manager.append_message("user", initial_user_prompt)

    # Generate and display the initial user message
    generate_impersonated_message_stream(
        api,
        conv_manager,
        model,
        roles,
        "user",
        options
    )

    while True:
        # Generate assistant message
        generate_impersonated_message_stream(
            api,
            conv_manager,
            model,
            roles,
            "assistant",
            options
        )
        time.sleep(delay)

        # Generate user message
        generate_impersonated_message_stream(
            api,
            conv_manager,
            model,
            roles,
            "user",
            options
        )
        time.sleep(delay)


def main():
    conv_manager = ConversationManager()
    api = OllamaAPI()

    console.print(
        "[bold magenta]Welcome to the Ollama Auto Chat CLI App![/bold magenta]"
    )

    # Check for existing conversations
    existing_convos = conv_manager.list_existing_conversations()
    selected_convo_folder = display_menu(existing_convos)

    if selected_convo_folder:
        # Load existing conversation and params
        conv_manager.load_conversation(selected_convo_folder)
        params_file = selected_convo_folder / "params.json"

        if not params_file.exists():
            console.print(
                f"[red]No params file found at {params_file}[/red]"
            )

            sys.exit(1)

        with params_file.open('r', encoding='utf-8') as f:
            params = json.load(f)

        selected_model = params.get("model")
        roles = params.get("roles")
        options = params.get("options", {})

        # Display the previous conversation for continuity
        conv_manager.display_conversation(roles)

    else:
        # Start a new conversation
        console.print("Fetching available models...\n")
        models = api.list_local_models()
        selected_model = api.select_model(models)

        # Define roles
        roles = get_role_descriptions()

        # Add system messages based on roles
        conv_manager.append_message(
            "system",
            roles["user"]["description"]
        )
        conv_manager.append_message(
            "system",
            roles["assistant"]["description"]
        )

        # Optional: Define custom options if needed
        options = {}
        if Confirm.ask("\nWould you like to set custom options for the model?", default=False):
            try:
                temperature = float(
                    Prompt.ask(
                        "Enter temperature (default=0.7)",
                        default="0.7"
                    )
                )
                max_tokens = int(
                    Prompt.ask(
                        "Enter max_tokens (default=150)",
                        default="150"
                    )
                )
                options["temperature"] = temperature
                options["max_tokens"] = max_tokens
            except ValueError:
                console.print(
                    "[red]Invalid input for options. Using default settings.[/red]"
                )

        # Create new conversation folder
        selected_convo_folder = conv_manager.create_new_conversation_folder()

        # Save initial parameters
        params = {
            "model": selected_model,
            "roles": roles,
            "options": options
        }
        with (selected_convo_folder / "params.json").open(
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(
                params,
                f,
                ensure_ascii=False,
                indent=4
            )

        console.print(
            f"[green]Parameters saved to {selected_convo_folder / 'params.json'}[/green]"
        )

    # Choose mode
    mode = select_mode()

    if mode == '1':
        # Interactive Chat Mode
        console.print(
            "\n[bold green]Starting Interactive Chat Mode[/bold green]"
        )

        stop_event = threading.Event()
        input_thread = threading.Thread(
            target=get_user_input,
            args=(
                api,
                conv_manager,
                selected_model,
                roles,
                options,
                selected_convo_folder,
                stop_event
            ),
            daemon=True
        )
        input_thread.start()

        try:
            while input_thread.is_alive():
                input_thread.join(timeout=1)

        except KeyboardInterrupt:
            stop_event.set()

            console.print(
                "\n[red]Exiting chat. Goodbye![/red]"
            )

            conv_manager.save_conversation(selected_convo_folder)

            sys.exit(0)

    else:
        # Automated Conversation Mode
        console.print(
            "\n[bold blue]Starting Automated Conversation Mode[/bold blue]"
        )
        try:
            auto_conversation(
                api,
                conv_manager,
                selected_model,
                roles,
                options
            )
            conv_manager.save_conversation(selected_convo_folder)

        except KeyboardInterrupt:
            console.print(
                "\n[red]Exiting automated conversation. Goodbye![/red]"
            )

            conv_manager.save_conversation(selected_convo_folder)

            sys.exit(0)

    # Save conversation on exit
    conv_manager.save_conversation(selected_convo_folder)


if __name__ == "__main__":
    main()
