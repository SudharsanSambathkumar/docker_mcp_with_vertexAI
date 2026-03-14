# Docker MCP Agent

An **AI-powered DevOps automation agent** that manages Docker infrastructure using the **Model Context Protocol (MCP)** and **Gemini models**.
The system allows an LLM to control Docker environments safely using structured tools.

It includes:

* An **MCP Docker server** exposing Docker operations as tools
* An **AI agent client** that decides which tools to call
* Safety checks for destructive actions
* Parallel tool execution and retries

---

# Architecture

```
User
  │
  ▼
AI Agent (client.py)
  │
  │  MCP Protocol
  ▼
Docker MCP Server (server.py)
  │
  ▼
Docker Engine
```

Components:

| Component          | Description                      |
| ------------------ | -------------------------------- |
| `server.py`        | MCP server exposing Docker tools |
| `client.py`        | AI DevOps agent using Gemini     |
| `requirements.txt` | Python dependencies              |

---

# Features

### AI Infrastructure Operator

The agent acts like a **DevOps engineer**, capable of managing Docker environments through natural language commands.

Example:

```
You: list all containers
You: pull nginx image
You: run nginx container on port 8080
You: show docker disk usage
```

---

### Docker Management Tools

The MCP server exposes multiple Docker capabilities:

#### Image Operations

* List images
* Pull image
* Push image
* Build image from Dockerfile
* Remove images
* Inspect image
* Tag image
* View image history
* Prune unused images

#### Container Operations

* List containers
* Run container
* Start / Stop / Restart
* Remove container
* Fetch container logs
* Execute commands inside containers
* Inspect container
* View container stats
* Copy files into containers

#### Network Operations

* List networks
* Create network
* Remove network
* Connect container to network

#### Volume Operations

* List volumes
* Create volume
* Remove volume

#### System Operations

* Docker system info
* Docker disk usage
* System prune
* Run docker-compose

---

# Safety Features

To prevent accidental infrastructure damage:

* Destructive commands require **manual confirmation**
* Tool execution has **timeouts**
* Automatic **retry mechanism**
* Parallel execution for read-only operations

Destructive actions include:

```
remove
delete
prune
force
stop
rm
```

---

# Requirements

Python **3.10+**

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies from :

```
mcp >= 1.5
docker >= 7.1
anyio >= 4.7
httpx >= 0.28
```

---

# Project Structure

```
project/
│
├── server.py        # MCP Docker server
├── client.py        # AI DevOps agent
├── requirements.txt
└── README.md
```

* `server.py` exposes Docker operations as MCP tools 
* `client.py` runs the AI agent loop and executes tools via MCP 

---

# Running the System

### 1 Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2 Ensure Docker is running

Test:

```bash
docker ps
```

---

### 3 Run the agent

```bash
python client.py
```

You should see:

```
Enterprise Docker Agent Ready.
Type 'exit' to quit.
```
### 4 Run Streamlit UI

```bash
streamlit run app.py
```
---

# Example Commands

```
list docker images

pull nginx image

run nginx container on port 8080

show docker disk usage

create docker network called backend

inspect container nginx
```

---

# Example Dockerfile Build

The agent can build images directly:

```
build docker image called sample-api with a fastapi dockerfile
```

The MCP server converts the Dockerfile text into a **temporary build context** before building the image.

---

# AI Model Configuration

The client currently uses:

```
MODEL_NAME = "gemini-2.5-flash"
```

You can replace it with:

* Gemini Pro
* OpenAI GPT
* Local LLMs
* Any MCP-compatible model

---

# Safety Confirmation Example

```
⚠ Destructive action detected (docker_container_remove).
Proceed? (yes/no):
```

This prevents accidental deletion of containers or images.

---

# Future Improvements

Potential enhancements:

* Kubernetes MCP tools
* Docker Swarm orchestration
* GPU container management
* Persistent agent memory
* Web UI dashboard
* CI/CD pipeline integration

---

# Use Cases

* AI DevOps assistants
* Autonomous infrastructure management
* Local container orchestration
* LLM-powered platform engineering
* Intelligent Docker operations
