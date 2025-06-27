<div align="center">

  ![lichess-bot](https://github.com/lichess-bot-devs/lichess-bot-images/blob/main/lichess-bot-icon-400.png)

  <h1>lichess-bot with LLM Integration</h1>

  A bridge between [lichess.org](https://lichess.org) and bots, featuring a Large Language Model (LLM) chess engine.
  <br>
  <strong>[Explore lichess-bot docs »](https://github.com/lichess-bot-devs/lichess-bot/wiki)</strong>
  <br>
  <br>
  [![Python Build](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml)
  [![Python Test](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml)
  [![Mypy](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/mypy.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/mypy.yml)

</div>

## Overview

This repository extends the [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) project by integrating a Large Language Model (LLM) as a chess engine. This project aims to test the ability of GPT models to play chess and provides a framework for running an LLM-powered chess bot on lichess.org.

With lichess-bot, you can create and operate a bot on lichess. Your bot will be able to play against humans and bots alike, and you will be able to view these games live on lichess.

See also the lichess-bot [documentation](https://github.com/lichess-bot-devs/lichess-bot/wiki) for further usage help.

## Features
Supports:
- Every variant and time control
- UCI, XBoard, and Homemade engines (including the new LLM engine)
- Matchmaking (challenging other bots)
- Offering Draws and Resigning
- Participating in tournaments
- Accepting move takeback requests from opponents
- Saving games as PGN
- Local & Online Opening Books
- Local & Online Endgame Tablebases

Can run on:
- Python 3.9 and later
- Windows, Linux and MacOS
- Docker

## Getting Started with the LLM Chess Bot

To run the LLM chess bot, follow these steps:

### 1. Install lichess-bot

Follow the standard installation instructions for your operating system:

*   **Linux**:
    ```bash
    # Download the repo
    git clone https://github.com/lichess-bot-devs/lichess-bot.git
    cd lichess-bot

    # Install dependencies
    sudo apt install python3 python3-pip python3-virtualenv python3-venv # Adjust for your package manager if not Ubuntu

    # Setup virtual environment
    python3 -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt
    ```

*   **Mac/BSD**:
    ```bash
    # Install Python and virtualenv using Homebrew
    brew install python3 virtualenv

    # Download the repo
    git clone https://github.com/lichess-bot-devs/lichess-bot.git
    cd lichess-bot

    # Setup virtual environment
    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
    ```

*   **Windows**:
    1.  [Download and install Python 3.9 or later](https://www.python.org/downloads/). Ensure "Add Python to PATH" is enabled during installation.
    2.  Open your preferred command prompt (Terminal, PowerShell, or cmd).
    3.  Upgrade pip: `py -m pip install --upgrade pip`.
    4.  Download the repo: `git clone https://github.com/lichess-bot-devs/lichess-bot.git`.
    5.  Navigate to the directory: `cd lichess-bot`.
    6.  Install virtualenv: `py -m pip install virtualenv`.
    7.  Setup virtualenv:
        ```cmd
        py -m venv venv
        venv\Scripts\activate
        pip install -r requirements.txt
        ```
        *(PowerShell note: If `activate` doesn't work, run `Set-ExecutionPolicy RemoteSigned` as administrator, then `Set-ExecutionPolicy Restricted` afterwards.)*

*   **Docker**:
    If you have a [Docker](https://www.docker.com/) host, you can use the `lichess-bot-devs/lichess-bot` [image in DockerHub](https://hub.docker.com/r/lichessbotdevs/lichess-bot).

### 2. Create a Lichess OAuth Token

Follow the instructions on the [Lichess Wiki](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token) to create an OAuth token for your bot.

### 3. Configure the LLM Engine

1.  **Copy `config.yml.default` to `config.yml`**:
    ```bash
    cp config.yml.default config.yml
    ```

2.  **Edit `config.yml`**:
    Open `config.yml` in a text editor and configure the `engine` section to use the LLM engine.

    ```yaml
    # Lichess API token. Required.
    # Example: "xxxxxxxxxxxxxxxx"
    lichess.token: "YOUR_LICHESS_API_TOKEN"

    # Lichess user ID. Required.
    # Example: "john-doe"
    lichess.user_id: "YOUR_LICHESS_USER_ID"

    # Engine protocol. Required.
    # Can be "uci", "xboard", or "homemade".
    # Example: "uci"
    engine.protocol: "uci"

    # Engine name. Required.
    # If protocol is "uci" or "xboard", this is the path to the engine executable.
    # If protocol is "homemade", this is the name of the class in homemade.py.
    # Example: "stockfish"
    engine.name: "llm_uci_engine.py"

    # Engine directory. Required if protocol is "uci" or "xboard".
    # This is the directory where the engine executable is located.
    # Example: "./engines"
    engine.dir: "."

    # Initial engine commands. Optional.
    # These commands are sent to the engine after it starts.
    # Example: ["setoption name Hash value 128", "setoption name Threads value 4"]
    engine.init_commands: [
      "setoption name LLM_Type value openai",
      "setoption name Model_Name value gpt-4o" # Or your preferred OpenAI model
    ]
    ```

    **Important**:
    *   Replace `"YOUR_LICHESS_API_TOKEN"` with the token you generated in step 2.
    *   Replace `"YOUR_LICHESS_USER_ID"` with your Lichess user ID.
    *   `engine.protocol` should be `uci`.
    *   `engine.name` should be `llm_uci_engine.py`.
    *   `engine.dir` should be `.` (dot) as `llm_uci_engine.py` is in the root directory.
    *   `engine.init_commands` is crucial for setting the LLM type and model. Currently, only `openai` is supported.

### 4. Set up OpenAI API Key

The LLM engine requires an OpenAI API key. Create a `.env` file in the root of your `lichess-bot` directory (the same directory as `lichess-bot.py`) and add your OpenAI API key:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

### 5. Upgrade to a BOT Account

Follow the instructions on the [Lichess Wiki](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account) to convert your Lichess account to a bot account.

### 6. Run lichess-bot

*   **Run from the command line**:
    1.  Open a command prompt or terminal.
    2.  Navigate to the `lichess-bot` directory.
    3.  Activate the virtual environment:
        *   Linux/Mac: `source venv/bin/activate`
        *   Windows: `venv\Scripts\activate`
    4.  Run lichess-bot: `python3 lichess-bot.py`

*   **Run with Docker**:
    ```bash
    docker run -it -v $(pwd)/config.yml:/usr/src/app/config.yml -v $(pwd)/.env:/usr/src/app/.env lichess-bot-devs/lichess-bot
    ```
    *(Note: You might need to build the Docker image first if it's not available on DockerHub with the LLM changes.)*

*   **Run with Docker Compose**:
    If you have a `docker-compose.yml` file configured, you can run:
    ```bash
    docker-compose up
    ```

*   **Other methods**:
    For other running methods (systemd, PM2, Windows Service, macOS Launch Agent), refer to the original [How to Run lichess-bot Wiki page](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Run-lichess%E2%80%90bot). Remember to adjust paths and ensure the `.env` file is accessible if using these methods.

## Troubleshooting
- If you get an error like `ModuleNotFoundError: No module named 'chess'`, make sure your virtual environment is activated and all dependencies are installed (`pip install -r requirements.txt`).
- If the bot doesn't respond to challenges, check your `config.yml` settings, especially the `challenge` section.
- Check the `llm_uci_engine.log` file and `lichess-bot.log` file for any errors or warnings.
- Ensure your `OPENAI_API_KEY` in the `.env` file is correct and has the necessary permissions.

## Acknowledgements
Thanks to the Lichess team, especially T. Alexander Lystad and Thibault Duplessis for working with the LeelaChessZero team to get this API up. Thanks to the [Niklas Fiekas](https://github.com/niklasf) and his [python-chess](https://github.com/niklasf/python-chess) code which allows engine communication seamlessly.

## License
lichess-bot is licensed under the AGPLv3 (or any later version at your option). Check out the [LICENSE file](https://github.com/lichess-bot-devs/lichess-bot/blob/master/LICENSE) for the full text.

## Citation
If this software has been used for research purposes, please cite it using the "Cite this repository" menu on the right sidebar. For more information, check the [CITATION file](https://github.com/lichess-bot-devs/lichess-bot/blob/master/CITATION.cff).