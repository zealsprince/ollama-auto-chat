
# Ollama Auto Chat

![Python Version](https://img.shields.io/badge/Python-3.13.1-blue.svg)

## Overview

**ollama-auto-chat** is a Python-based command-line interface (CLI) application that enables automated conversations between chat agents using the Ollama API. Whether you're looking to simulate dialogues, test conversational flows, or simply experiment with AI-driven interactions, this tool provides a straightforward playground.

## Features (That differ from just Ollama)

- **Automated Conversation Mode:** Allow agents to converse autonomously.
- **Conversation History:** Save and load past conversations for continuity.
- **Rich Text Interface:** Enhanced command-line experience with colored and formatted text using the Rich library.

## Installation

1. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt`, you can install the necessary packages manually:*

   ```bash
   pip install requests rich
   ```

3. **Configure Ollama API**

   Ensure that the Ollama API is running and accessible at `http://localhost:11434`. You can modify the `OLLAMA_HOST` variable in the script if your API is hosted elsewhere.

## Usage

Run the application using Python:

```bash
python ollama_auto_chat.py
```

Follow the on-screen prompts to:

- Select or create a conversation.
- Choose a chat model.
- Define roles and system prompts.
- Select between Interactive Chat Mode or Automated Conversation Mode.

*Type `exit` or `quit` at any prompt to gracefully exit the application.*

## License

This project is licensed under the [MIT License](LICENSE).
