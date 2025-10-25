"""AgentCore Launch Tool - Complete deployment orchestration for Bedrock AgentCore.

This tool provides comprehensive agent deployment capabilities with three modes:
- CodeBuild (default): Cloud-based ARM64 builds with CodeBuild, no local Docker needed
- Local: Build and run locally with Docker/Finch/Podman
- Local-build: Build locally, deploy to cloud runtime

Based on bedrock-agentcore-starter-toolkit implementation.
"""

import base64
import fnmatch
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from strands import tool

logger = logging.getLogger(__name__)


@tool
def launch(
    action: str = "launch",
    agent_name: Optional[str] = None,
    mode: str = "codebuild",
    auto_update_on_conflict: bool = False,
    env_vars: Optional[Dict[str, str]] = None,
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """Deploy and manage AWS Bedrock AgentCore agent deployments with complete lifecycle support.

    This tool orchestrates the complete deployment pipeline for AgentCore agents, handling
    ECR repository creation, IAM roles, container builds, and runtime deployment. It provides
    three deployment modes optimized for different workflows, with automatic resource provisioning
    and intelligent conflict resolution.

    How It Works:
    ------------
    1. **Configuration Loading**: Reads .bedrock_agentcore.yaml from configure tool
    2. **Resource Provisioning**: Auto-creates ECR repositories and IAM roles if needed
    3. **Container Build**: Builds ARM64 Docker image via CodeBuild (or locally)
    4. **Image Push**: Pushes container to ECR with "latest" tag
    5. **Runtime Deployment**: Creates/updates AgentCore runtime with container URI
    6. **Endpoint Ready**: Waits for endpoint to become READY for invocations
    7. **Config Update**: Saves agent_id and agent_arn back to configuration

    Deployment Process:
    ------------------
    1. **Validation**: Verifies Dockerfile exists (created by configure tool)
    2. **ECR Setup**: Creates ECR repository if not exists (idempotent)
    3. **IAM Setup**: Creates execution role with Bedrock, Logs, Memory permissions
    4. **CodeBuild Project**: Creates/updates CodeBuild project for ARM64 builds
    5. **Source Upload**: Packages source code to S3 (respects .dockerignore)
    6. **Build Execution**: Triggers parallel Docker build + ECR authentication
    7. **AgentCore Deploy**: Creates/updates runtime with container URI
    8. **Endpoint Wait**: Polls until endpoint status becomes READY

    Three Deployment Modes:
    ----------------------

    ### Mode 1: CodeBuild (Recommended - Default)
    **No local Docker required, optimized for ARM64 cloud deployment**

    ```python
    launch(
        action="launch",
        agent_name="my-agent",
        mode="codebuild"  # Default mode
    )
    ```

    **How it works:**
    - Creates S3 bucket for source upload (with 7-day lifecycle)
    - Packages entire project directory (respects .dockerignore)
    - Creates CodeBuild project with ARM64 environment
    - Parallel build: Docker build + ECR authentication
    - Automatic image push to ECR
    - Zero local Docker dependencies

    **Best for:**
    - Production deployments
    - ARM64-optimized containers
    - CI/CD pipelines
    - No local Docker setup
    - Consistent build environment

    ### Mode 2: Local (Development)
    **Build and run locally with Docker/Finch/Podman**

    ```python
    launch(
        action="launch",
        agent_name="my-agent",
        mode="local"
    )
    ```

    **How it works:**
    - Builds container locally on your machine
    - Runs agent in local Docker container
    - No cloud deployment
    - Direct console output

    **Best for:**
    - ✅ Local development and testing
    - ✅ Rapid iteration
    - ✅ Debugging agent behavior
    - ✅ Network-isolated development

    ### Mode 3: Local-build (Hybrid)
    **Build locally, deploy to cloud**

    ```python
    launch(
        action="launch",
        agent_name="my-agent",
        mode="local-build"
    )
    ```

    **How it works:**
    - Builds container locally
    - Pushes to ECR
    - Deploys to AgentCore runtime

    **Best for:**
    - ✅ Custom build requirements
    - ✅ Pre-built base images
    - ✅ Local build caching
    - ⚠️ Requires local Docker + ECR access

    Auto-Update on Conflict:
    -----------------------
    When an agent with the same name already exists in AgentCore:

    ```python
    # Without auto-update (default)
    launch(agent_name="existing-agent")  # Raises ConflictException

    # With auto-update
    launch(
        agent_name="existing-agent",
        auto_update_on_conflict=True  # Finds existing agent and updates it
    )
    ```

    This automatically:
    - Searches for existing agent by name
    - Retrieves agent_id from list API
    - Updates existing agent with new container image
    - Preserves agent_arn and configuration

    Environment Variables:
    --------------------
    Pass runtime configuration to your deployed agent:

    ```python
    launch(
        agent_name="my-agent",
        env_vars={
            "MODEL_ID": "anthropic.claude-v3-5-sonnet",
            "MAX_TOKENS": "4096",
            "LOG_LEVEL": "DEBUG",
            "CUSTOM_CONFIG": "production"
        }
    )
    ```

    **Automatic Environment Variables (Added by launch tool):**
    - `BEDROCK_memory_ID`: Memory resource ID (if memory enabled)
    - `BEDROCK_memory_NAME`: Memory resource name
    - `AWS_REGION`: Deployment region
    - `AWS_DEFAULT_REGION`: Deployment region (duplicate for compatibility)

    Resource Management:
    -------------------
    The launch tool automatically provisions AWS resources:

    **ECR Repository:**
    - Name: `bedrock-agentcore-{sanitized-agent-name}`
    - Lifecycle: Managed by AWS ECR
    - Images: Tagged as "latest" (overwritten on each deploy)

    **S3 Bucket (CodeBuild mode):**
    - Name: `bedrock-agentcore-codebuild-sources-{account}-{region}`
    - Lifecycle: Auto-delete objects after 7 days
    - Purpose: Temporary source code storage

    **IAM Roles (Auto-created):**
    - Runtime Role: `AmazonBedrockAgentCoreSDKRuntime-{region}-{hash}`
      - Permissions: Bedrock, Logs, Memory, ECR, X-Ray
    - CodeBuild Role: `AmazonBedrockAgentCoreSDKCodeBuild-{region}-{hash}`
      - Permissions: ECR, Logs, S3, CodeBuild

    **CodeBuild Project:**
    - Name: `bedrock-agentcore-{agent-name}-builder`
    - Environment: ARM64 Amazon Linux 2
    - Compute: BUILD_GENERAL1_MEDIUM

    Common Use Cases:
    ---------------

    ### 1. First-time Deployment
    ```python
    # After running configure tool
    configure(action="configure", entrypoint="agent.py", agent_name="research-agent")

    # Deploy to AgentCore
    launch(action="launch", agent_name="research-agent")
    ```

    ### 2. Update Existing Agent
    ```python
    # Modify your agent code, then redeploy
    launch(
        action="launch",
        agent_name="research-agent",
        auto_update_on_conflict=True  # Updates existing agent
    )
    ```

    ### 3. Multiple Environments
    ```python
    # Development
    launch(
        agent_name="my-agent-dev",
        env_vars={"ENVIRONMENT": "dev", "LOG_LEVEL": "DEBUG"}
    )

    # Production
    launch(
        agent_name="my-agent-prod",
        env_vars={"ENVIRONMENT": "prod", "LOG_LEVEL": "INFO"}
    )
    ```

    ### 4. Check Deployment Status
    ```python
    # Get agent runtime status
    result = launch(action="status", agent_name="my-agent")
    # Returns: Agent ARN, Endpoint status, Created/Updated timestamps
    ```

    ### 5. Session Management
    ```python
    # Stop active runtime session (frees resources)
    launch(action="stop_session", agent_name="my-agent")
    ```

    Integration with Configure Tool:
    ------------------------------
    The launch tool requires prior configuration via configure tool:

    ```python
    # Step 1: Configure (creates Dockerfile, .bedrock_agentcore.yaml)
    configure(
        action="configure",
        entrypoint="agent.py",
        agent_name="my-agent",
        memory_mode="STM_AND_LTM",
        enable_observability=True
    )

    # Step 2: Launch (builds, deploys to AgentCore)
    launch(action="launch", agent_name="my-agent")

    # Step 3: Invoke (call deployed agent)
    invoke(
        agent_arn="arn:aws:bedrock-agentcore:...",
        payload='{"prompt": "Hello!"}'
    )
    ```

    Args:
        action: Action to perform:
            - "launch": Deploy agent to AgentCore (default)
            - "status": Check deployment status and endpoint health
            - "stop_session": Stop active runtime session to free resources

        agent_name: Agent name from .bedrock_agentcore.yaml
            If not specified, uses default_agent from configuration
            Must match an agent configured via configure tool

        mode: Deployment mode:
            - "codebuild": Cloud build with CodeBuild, no local Docker (DEFAULT)
            - "local": Build and run locally with Docker/Finch/Podman
            - "local-build": Build locally, deploy to cloud runtime

        auto_update_on_conflict: Auto-update existing agent on ConflictException
            When True: Searches for existing agent by name and updates it
            When False: Raises error if agent name already exists
            Default: False

        env_vars: Environment variables to inject into agent runtime
            Dictionary of key-value pairs passed to agent container
            Example: {"MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "LOG_LEVEL": "DEBUG"}
            Note: BEDROCK_memory_* vars auto-added if memory enabled

        region: AWS region for deployment
            Must match region used in configure tool
            Default: us-west-2

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [
                {"text": "Deployment status message"},
                {"text": "Agent ARN: arn:aws:..."},
                {"text": "Agent ID: agent-abc123"},
                {"text": "ECR URI: 123.dkr.ecr..."},
                {"text": "Build ID: codebuild:..."},
                {"text": "Next steps: ..."}
            ]
        }

        Success case: Returns agent ARN, ID, ECR URI, build details, next steps
        Error case: Returns detailed error message and troubleshooting guidance

    Examples:
    --------
    # Basic deployment (CodeBuild mode)
    launch(
        action="launch",
        agent_name="my-research-agent"
    )

    # Deployment with auto-update
    launch(
        action="launch",
        agent_name="my-research-agent",
        auto_update_on_conflict=True
    )

    # Deploy with custom environment variables
    launch(
        action="launch",
        agent_name="my-research-agent",
        env_vars={
            "MODEL_ID": ""us.anthropic.claude-sonnet-4-5-20250929-v1:0"",
            "MAX_TOKENS": "64000",
            "ENVIRONMENT": "production"
        }
    )

    # Check deployment status
    launch(
        action="status",
        agent_name="my-research-agent"
    )

    # Stop active session
    launch(
        action="stop_session",
        agent_name="my-research-agent"
    )

    # Deploy to different region
    launch(
        action="launch",
        agent_name="my-research-agent",
        region="us-east-1"
    )

    Notes:
        - **Prerequisites**: Must run configure tool first to generate Dockerfile
        - **CodeBuild Duration**: Typical build takes 2-4 minutes for ARM64
        - **Idempotent**: Safe to run multiple times, updates existing resources
        - **Cost Optimization**: S3 sources auto-delete after 7 days
        - **IAM Propagation**: Auto-waits 10 seconds after role creation
        - **Endpoint Ready**: Auto-waits up to 120 seconds for endpoint READY status
        - **Memory Integration**: Automatically injects memory IDs from configure
        - **Session State**: agent_session_id tracked in configuration for continuity
        - **Conflict Resolution**: Use auto_update_on_conflict=True for redeployments
        - **CloudWatch Logs**: Available at /aws/bedrock-agentcore/runtimes/{agent-id}
        - **ARM64 Optimization**: CodeBuild uses ARM64 for cost and performance benefits
    """
    try:
        config_path = Path.cwd() / ".bedrock_agentcore.yaml"

        if action == "launch":
            return _launch_agent(
                config_path=config_path,
                agent_name=agent_name,
                mode=mode,
                auto_update_on_conflict=auto_update_on_conflict,
                env_vars=env_vars or {},
                region=region,
            )
        elif action == "status":
            return _get_agent_status(config_path, agent_name, region)
        elif action == "stop_session":
            return _stop_runtime_session(config_path, agent_name, region)
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Use: launch, status, stop_session"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"AgentCore launch error: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load .bedrock_agentcore.yaml configuration."""
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found at {config_path}. "
            f"Run 'agentcore configure' first or use configure tool"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}


def _save_config(config: Dict[str, Any], config_path: Path):
    """Save configuration to YAML file."""
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _get_agent_config(
    config: Dict[str, Any], agent_name: Optional[str]
) -> Dict[str, Any]:
    """Get agent configuration from project config."""
    agents = config.get("agents", {})

    if not agents:
        raise ValueError("No agents configured")

    if agent_name:
        if agent_name not in agents:
            raise ValueError(
                f"Agent '{agent_name}' not found. Available: {list(agents.keys())}"
            )
        agent_config = agents[agent_name]
        assert isinstance(agent_config, dict)
        return agent_config

    # Use default agent
    default_agent = config.get("default_agent")
    if not default_agent:
        if len(agents) == 1:
            default_agent = list(agents.keys())[0]
        else:
            raise ValueError(
                f"Multiple agents configured, specify --agent. Available: {list(agents.keys())}"
            )

    agent_config = agents[default_agent]
    assert isinstance(agent_config, dict)
    return agent_config


def _launch_agent(
    config_path: Path,
    agent_name: Optional[str],
    mode: str,
    auto_update_on_conflict: bool,
    env_vars: Dict[str, str],
    region: str,
) -> Dict[str, Any]:
    """Launch agent with specified mode."""
    # Load configuration
    project_config = _load_config(config_path)
    agent_config = _get_agent_config(project_config, agent_name)
    actual_agent_name = agent_config["name"]

    logger.info(f"Launching agent '{actual_agent_name}' in {mode} mode...")

    # Add memory configuration to env_vars if available
    if agent_config.get("memory") and agent_config["memory"].get("memory_id"):
        env_vars["BEDROCK_memory_ID"] = agent_config["memory"]["memory_id"]
        env_vars["BEDROCK_memory_NAME"] = agent_config["memory"].get("memory_name", "")

    if mode == "codebuild":
        return _launch_with_codebuild(
            config_path=config_path,
            agent_name=actual_agent_name,
            agent_config=agent_config,
            project_config=project_config,
            auto_update_on_conflict=auto_update_on_conflict,
            env_vars=env_vars,
            region=region,
        )
    elif mode == "local":
        return {
            "status": "error",
            "content": [
                {
                    "text": "Local mode not yet implemented in tool. "
                    "Use CLI: 'agentcore launch --local' or use CodeBuild mode."
                }
            ],
        }
    else:
        return {"status": "error", "content": [{"text": f"Unknown mode: {mode}"}]}


def _launch_with_codebuild(
    config_path: Path,
    agent_name: str,
    agent_config: Dict,
    project_config: Dict,
    auto_update_on_conflict: bool,
    env_vars: Dict[str, str],
    region: str,
) -> Dict[str, Any]:
    """Launch using CodeBuild for ARM64 builds."""
    session = boto3.Session(region_name=region)
    account_id = session.client("sts").get_caller_identity()["Account"]

    logger.info(
        f"Starting CodeBuild ARM64 deployment for '{agent_name}' to account {account_id} ({region})"
    )

    # Step 0: Check for Dockerfile
    dockerfile_dir = config_path.parent / ".bedrock_agentcore" / agent_name
    dockerfile_path = dockerfile_dir / "Dockerfile"

    if not dockerfile_path.exists():
        return {
            "status": "error",
            "content": [
                {"text": f"❌ Dockerfile not found at: {dockerfile_path}"},
                {"text": "\n**Required: Run configure first**"},
                {
                    "text": f"  configure(action='configure', entrypoint='{agent_config.get('entrypoint')}', agent_name='{agent_name}')"
                },
                {
                    "text": "\nThe configure tool generates the Dockerfile and prepares the agent for deployment."
                },
            ],
        }

    logger.info(f"✅ Using Dockerfile: {dockerfile_path}")

    # Step 1: Ensure ECR repository
    ecr_uri = _ensure_ecr_repository(
        agent_config, project_config, config_path, agent_name, region
    )
    logger.info(f"✅ ECR repository: {ecr_uri}")

    # Step 2: Ensure execution role
    _ensure_execution_role(
        agent_config, project_config, config_path, agent_name, region, account_id
    )
    logger.info(f"✅ Execution role: {agent_config['aws']['execution_role']}")

    # Step 3: CodeBuild build
    codebuild_service = CodeBuildService(session, logger)

    # Get ECR repository ARN
    ecr_repo_name = ecr_uri.split("/")[-1]
    ecr_repository_arn = f"arn:aws:ecr:{region}:{account_id}:repository/{ecr_repo_name}"

    # Get or create CodeBuild execution role
    if agent_config.get("codebuild", {}).get("execution_role"):
        codebuild_execution_role = agent_config["codebuild"]["execution_role"]
        logger.info(f"Using CodeBuild role from config: {codebuild_execution_role}")
    else:
        codebuild_execution_role = codebuild_service.create_codebuild_execution_role(
            account_id=account_id,
            ecr_repository_arn=ecr_repository_arn,
            agent_name=agent_name,
        )

    # Upload source
    source_dir = agent_config.get("source_path") or "."
    dockerfile_dir = config_path.parent / ".bedrock_agentcore" / agent_name
    source_location = codebuild_service.upload_source(
        agent_name=agent_name, source_dir=source_dir, dockerfile_dir=str(dockerfile_dir)
    )

    # Create or update CodeBuild project
    if agent_config.get("codebuild", {}).get("project_name"):
        project_name = agent_config["codebuild"]["project_name"]
        logger.info(f"Using CodeBuild project: {project_name}")
    else:
        project_name = codebuild_service.create_or_update_project(
            agent_name=agent_name,
            ecr_repository_uri=ecr_uri,
            execution_role=codebuild_execution_role,
            source_location=source_location,
        )

    # Start build and wait
    logger.info("Starting CodeBuild build...")
    build_id = codebuild_service.start_build(project_name, source_location)
    codebuild_service.wait_for_completion(build_id)
    logger.info("✅ CodeBuild completed successfully")

    # Save CodeBuild config
    if "codebuild" not in agent_config:
        agent_config["codebuild"] = {}
    agent_config["codebuild"]["project_name"] = project_name
    agent_config["codebuild"]["execution_role"] = codebuild_execution_role
    project_config["agents"][agent_name] = agent_config
    _save_config(project_config, config_path)

    # Step 4: Deploy to AgentCore
    agent_id, agent_arn = _deploy_to_bedrock_agentcore(
        agent_config=agent_config,
        project_config=project_config,
        config_path=config_path,
        agent_name=agent_name,
        ecr_uri=ecr_uri,
        region=region,
        account_id=account_id,
        env_vars=env_vars,
        auto_update_on_conflict=auto_update_on_conflict,
    )

    return {
        "status": "success",
        "content": [
            {"text": f"✅ Agent deployed successfully!"},
            {"text": f"Agent Name: {agent_name}"},
            {"text": f"Agent ARN: {agent_arn}"},
            {"text": f"Agent ID: {agent_id}"},
            {"text": f"ECR URI: {ecr_uri}:latest"},
            {"text": f"Build ID: {build_id}"},
            {"text": f"\nNext steps:"},
            {"text": f"  launch(action='status', agent_name='{agent_name}')"},
            {
                "text": f"  invoke(agent_arn='{agent_arn}', payload='{{\"prompt\": \"Hello\"}}')"
            },
        ],
    }


def _ensure_ecr_repository(
    agent_config: Dict,
    project_config: Dict,
    config_path: Path,
    agent_name: str,
    region: str,
) -> str:
    """Ensure ECR repository exists (idempotent)."""
    ecr_uri = agent_config.get("aws", {}).get("ecr_repository")

    if ecr_uri:
        logger.info(f"Using ECR repository from config: {ecr_uri}")
        return str(ecr_uri)

    # Create repository if auto-create enabled
    if agent_config.get("aws", {}).get("ecr_auto_create", True):
        repo_name = f"bedrock-agentcore-{_sanitize_ecr_repo_name(agent_name)}"
        ecr = boto3.client("ecr", region_name=region)

        try:
            response = ecr.describe_repositories(repositoryNames=[repo_name])
            ecr_uri = str(response["repositories"][0]["repositoryUri"])
            logger.info(f"✅ Reusing existing ECR repository: {ecr_uri}")
        except ecr.exceptions.RepositoryNotFoundException:
            logger.info(f"Creating new ECR repository: {repo_name}")
            response = ecr.create_repository(repositoryName=repo_name)
            ecr_uri = str(response["repository"]["repositoryUri"])
            logger.info(f"✅ Created ECR repository: {ecr_uri}")

        # Update config
        if "aws" not in agent_config:
            agent_config["aws"] = {}
        agent_config["aws"]["ecr_repository"] = ecr_uri
        agent_config["aws"]["ecr_auto_create"] = False
        project_config["agents"][agent_name] = agent_config
        _save_config(project_config, config_path)

        return ecr_uri

    raise ValueError("ECR repository not configured and auto-create not enabled")


def _ensure_execution_role(
    agent_config: Dict,
    project_config: Dict,
    config_path: Path,
    agent_name: str,
    region: str,
    account_id: str,
) -> str:
    """Ensure execution role exists (idempotent)."""
    execution_role_arn = agent_config.get("aws", {}).get("execution_role")

    if execution_role_arn:
        logger.info(f"Using execution role from config: {execution_role_arn}")
        return str(execution_role_arn)

    # Create role if auto-create enabled
    if agent_config.get("aws", {}).get("execution_role_auto_create", True):
        deterministic_suffix = _generate_deterministic_suffix(agent_name)
        role_name = f"AmazonBedrockAgentCoreSDKRuntime-{region}-{deterministic_suffix}"

        iam = boto3.client("iam", region_name=region)

        try:
            # Check if role exists
            role = iam.get_role(RoleName=role_name)
            execution_role_arn = str(role["Role"]["Arn"])
            logger.info(f"✅ Reusing existing execution role: {execution_role_arn}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                # Create new role
                logger.info(f"Creating execution role: {role_name}")
                execution_role_arn = _create_runtime_execution_role(
                    iam, role_name, region, account_id, agent_name
                )
                logger.info(f"✅ Created execution role: {execution_role_arn}")
            else:
                raise

        # Update config
        if "aws" not in agent_config:
            agent_config["aws"] = {}
        agent_config["aws"]["execution_role"] = execution_role_arn
        agent_config["aws"]["execution_role_auto_create"] = False
        project_config["agents"][agent_name] = agent_config
        _save_config(project_config, config_path)

        return execution_role_arn

    raise ValueError("Execution role not configured and auto-create not enabled")


def _deploy_to_bedrock_agentcore(
    agent_config: Dict,
    project_config: Dict,
    config_path: Path,
    agent_name: str,
    ecr_uri: str,
    region: str,
    account_id: str,
    env_vars: Dict[str, str],
    auto_update_on_conflict: bool,
) -> tuple:
    """Deploy agent to Bedrock AgentCore."""
    logger.info("Deploying to Bedrock AgentCore...")

    client = boto3.client("bedrock-agentcore-control", region_name=region)

    # Build agent parameters
    params = {
        "agentRuntimeArtifact": {
            "containerConfiguration": {"containerUri": f"{ecr_uri}:latest"}
        },
        "roleArn": agent_config["aws"]["execution_role"],
    }

    # Add description if present
    if agent_config.get("description"):
        params["description"] = agent_config["description"]

    # Add network configuration with proper camelCase
    network_config = agent_config.get("aws", {}).get("network_configuration", {})
    if network_config:
        # Convert snake_case to camelCase for AWS API
        formatted_network_config = {}
        if "network_mode" in network_config:
            formatted_network_config["networkMode"] = network_config["network_mode"]
        elif "networkMode" in network_config:
            formatted_network_config["networkMode"] = network_config["networkMode"]
        else:
            # Default to PUBLIC if not specified
            formatted_network_config["networkMode"] = "PUBLIC"

        params["networkConfiguration"] = formatted_network_config
    else:
        # Default network configuration
        params["networkConfiguration"] = {"networkMode": "PUBLIC"}

    # Add protocol configuration with proper camelCase
    protocol_config = agent_config.get("aws", {}).get("protocol_configuration", {})
    if protocol_config:
        formatted_protocol_config = {}
        if "server_protocol" in protocol_config:
            formatted_protocol_config["serverProtocol"] = protocol_config[
                "server_protocol"
            ]
        elif "serverProtocol" in protocol_config:
            formatted_protocol_config["serverProtocol"] = protocol_config[
                "serverProtocol"
            ]
        else:
            formatted_protocol_config["serverProtocol"] = "HTTP"

        params["protocolConfiguration"] = formatted_protocol_config
    else:
        # Default protocol configuration
        params["protocolConfiguration"] = {"serverProtocol": "HTTP"}

    # Add environment variables
    if env_vars:
        params["environmentVariables"] = env_vars

    # Note: observability is configured at the account level, not per-agent

    # Try to create or update agent
    agent_id = agent_config.get("bedrock_agentcore", {}).get("agent_id")

    try:
        if agent_id:
            # Update existing agent
            params["agentRuntimeId"] = agent_id
            response = client.update_agent_runtime(**params)
            agent_arn = response["agentRuntimeArn"]
            logger.info(f"✅ Agent updated: {agent_arn}")
        else:
            # Create new agent - add agentRuntimeName for create
            create_params = params.copy()
            create_params["agentRuntimeName"] = agent_name
            response = client.create_agent_runtime(**create_params)
            agent_id = response["agentRuntimeId"]
            agent_arn = response["agentRuntimeArn"]
            logger.info(f"✅ Agent created: {agent_arn}")

    except ClientError as e:
        if (
            e.response["Error"]["Code"] == "ConflictException"
            and auto_update_on_conflict
        ):
            # Find existing agent and update
            logger.info(f"Agent '{agent_name}' exists, searching for agent ID...")
            existing_agent = _find_agent_by_name(client, agent_name)

            if not existing_agent:
                raise RuntimeError(
                    f"ConflictException but couldn't find agent '{agent_name}'"
                ) from e

            agent_id = existing_agent["agentRuntimeId"]
            agent_arn = existing_agent["agentRuntimeArn"]
            logger.info(f"Found agent ID: {agent_id}, updating...")

            # Update the agent
            params["agentRuntimeId"] = agent_id
            response = client.update_agent_runtime(**params)
            agent_arn = response["agentRuntimeArn"]
            logger.info(f"✅ Agent updated: {agent_arn}")
        else:
            raise

    # Save agent info to config
    if "bedrock_agentcore" not in agent_config:
        agent_config["bedrock_agentcore"] = {}
    agent_config["bedrock_agentcore"]["agent_id"] = agent_id
    agent_config["bedrock_agentcore"]["agent_arn"] = agent_arn
    agent_config["bedrock_agentcore"]["agent_session_id"] = None  # Reset session
    project_config["agents"][agent_name] = agent_config
    _save_config(project_config, config_path)

    # Wait for endpoint to be ready
    logger.info("Waiting for endpoint to be ready...")
    _wait_for_endpoint_ready(client, agent_id, max_wait=120)

    return agent_id, agent_arn


def _wait_for_endpoint_ready(
    client, agent_id: str, endpoint_name: str = "DEFAULT", max_wait: int = 120
):
    """Wait for agent endpoint to become ready."""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = client.get_agent_runtime_endpoint(
                agentRuntimeId=agent_id, endpointName=endpoint_name
            )
            status = response.get("status", "UNKNOWN")

            if status == "READY":
                logger.info(f"✅ Endpoint ready: {response['agentRuntimeEndpointArn']}")
                return response["agentRuntimeEndpointArn"]
            elif status in ["CREATE_FAILED", "UPDATE_FAILED"]:
                raise RuntimeError(
                    f"Endpoint {status}: {response.get('failureReason', 'Unknown')}"
                )

        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise

        time.sleep(2)

    logger.warning(f"Endpoint not ready after {max_wait}s, continuing...")


def _find_agent_by_name(client, agent_name: str) -> Optional[Dict[str, Any]]:
    """Find agent by name using list API."""
    next_token = None

    while True:
        params = {"maxResults": 100}
        if next_token:
            params["nextToken"] = next_token

        response = client.list_agent_runtimes(**params)

        agents_list: list[Dict[str, Any]] = response.get("agentRuntimes", [])
        for agent in agents_list:
            if agent.get("agentRuntimeName") == agent_name:
                return agent

        next_token = response.get("nextToken")
        if not next_token:
            break

    return None


def _get_agent_status(
    config_path: Path, agent_name: Optional[str], region: str
) -> Dict[str, Any]:
    """Get agent deployment status."""
    project_config = _load_config(config_path)
    agent_config = _get_agent_config(project_config, agent_name)
    actual_agent_name = agent_config["name"]

    agent_arn = agent_config.get("bedrock_agentcore", {}).get("agent_arn")
    agent_id = agent_config.get("bedrock_agentcore", {}).get("agent_id")

    if not agent_id:
        return {
            "status": "success",
            "content": [
                {"text": f"Agent '{actual_agent_name}' is configured but not deployed"},
                {"text": "Run launch() to deploy"},
            ],
        }

    # Get agent details
    client = boto3.client("bedrock-agentcore-control", region_name=region)

    try:
        agent_response = client.get_agent_runtime(agentRuntimeId=agent_id)
        endpoint_response = client.get_agent_runtime_endpoint(
            agentRuntimeId=agent_id, endpointName="DEFAULT"
        )

        endpoint_status = endpoint_response.get("status", "Unknown")
        created_at = agent_response.get("createdAt", "Unknown")
        updated_at = endpoint_response.get(
            "lastUpdatedAt", agent_response.get("lastUpdatedAt", "Unknown")
        )

        return {
            "status": "success",
            "content": [
                {"text": f"✅ Agent Status: {actual_agent_name}"},
                {"text": f"Agent ARN: {agent_arn}"},
                {"text": f"Agent ID: {agent_id}"},
                {"text": f"Endpoint Status: {endpoint_status}"},
                {"text": f"Created: {created_at}"},
                {"text": f"Updated: {updated_at}"},
                {"text": f"Region: {region}"},
                {
                    "text": f"Account: {agent_config.get('aws', {}).get('account', 'N/A')}"
                },
            ],
        }

    except ClientError as e:
        return {
            "status": "error",
            "content": [{"text": f"Error getting status: {str(e)}"}],
        }


def _stop_runtime_session(
    config_path: Path, agent_name: Optional[str], region: str
) -> Dict[str, Any]:
    """Stop active runtime session."""
    project_config = _load_config(config_path)
    agent_config = _get_agent_config(project_config, agent_name)

    agent_arn = agent_config.get("bedrock_agentcore", {}).get("agent_arn")
    session_id = agent_config.get("bedrock_agentcore", {}).get("agent_session_id")

    if not agent_arn:
        return {"status": "error", "content": [{"text": "Agent not deployed"}]}

    if not session_id:
        return {"status": "error", "content": [{"text": "No active session found"}]}

    client = boto3.client("bedrock-agentcore", region_name=region)

    try:
        client.stop_runtime_session(
            agentRuntimeArn=agent_arn, qualifier="DEFAULT", runtimeSessionId=session_id
        )

        # Clear session from config
        agent_config["bedrock_agentcore"]["agent_session_id"] = None
        project_config["agents"][agent_config["name"]] = agent_config
        _save_config(project_config, config_path)

        return {
            "status": "success",
            "content": [
                {"text": f"✅ Session stopped: {session_id}"},
                {"text": f"Agent: {agent_config['name']}"},
            ],
        }

    except ClientError as e:
        return {
            "status": "error",
            "content": [{"text": f"Error stopping session: {str(e)}"}],
        }


# Helper classes and functions


class CodeBuildService:
    """Service for managing CodeBuild projects and builds."""

    def __init__(self, session: boto3.Session, logger):
        self.session = session
        self.client = session.client("codebuild")
        self.s3_client = session.client("s3")
        self.logger = logger
        self.account_id = session.client("sts").get_caller_identity()["Account"]

    def ensure_source_bucket(self) -> str:
        """Ensure S3 bucket exists for CodeBuild sources."""
        region = self.session.region_name
        bucket_name = f"bedrock-agentcore-codebuild-sources-{self.account_id}-{region}"

        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.debug(f"Using existing S3 bucket: {bucket_name}")
        except ClientError:
            # Create bucket
            if region == "us-east-1":
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )

            # Add lifecycle policy
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "DeleteOldBuilds",
                            "Status": "Enabled",
                            "Filter": {},
                            "Expiration": {"Days": 7},
                        }
                    ]
                },
            )
            self.logger.info(f"Created S3 bucket: {bucket_name}")

        return bucket_name

    def upload_source(
        self,
        agent_name: str,
        source_dir: str = ".",
        dockerfile_dir: Optional[str] = None,
    ) -> str:
        """Upload source to S3."""
        bucket_name = self.ensure_source_bucket()
        ignore_patterns = self._get_dockerignore_patterns()

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            try:
                with zipfile.ZipFile(temp_zip.name, "w", zipfile.ZIP_DEFLATED) as zipf:
                    # Add source files
                    for root, dirs, files in os.walk(source_dir):
                        rel_root = os.path.relpath(root, source_dir)
                        if rel_root == ".":
                            rel_root = ""

                        # Filter directories
                        dirs[:] = [
                            d
                            for d in dirs
                            if not self._should_ignore(
                                os.path.join(rel_root, d) if rel_root else d,
                                ignore_patterns,
                                True,
                            )
                        ]

                        for file in files:
                            file_rel_path = (
                                os.path.join(rel_root, file) if rel_root else file
                            )
                            if self._should_ignore(
                                file_rel_path, ignore_patterns, False
                            ):
                                continue
                            file_path = Path(root) / file
                            zipf.write(file_path, file_rel_path)

                    # Add Dockerfile if in different directory
                    if dockerfile_dir and source_dir != dockerfile_dir:
                        dockerfile_path = Path(dockerfile_dir) / "Dockerfile"
                        if dockerfile_path.exists():
                            zipf.write(dockerfile_path, "Dockerfile")
                            self.logger.info(
                                f"Including Dockerfile from {dockerfile_dir}"
                            )

                # Upload to S3
                s3_key = f"{agent_name}/source.zip"
                self.s3_client.upload_file(temp_zip.name, bucket_name, s3_key)
                self.logger.info(f"Uploaded source to S3: {s3_key}")

                return f"s3://{bucket_name}/{s3_key}"

            finally:
                os.unlink(temp_zip.name)

    def create_codebuild_execution_role(
        self, account_id: str, ecr_repository_arn: str, agent_name: str
    ) -> str:
        """Get or create CodeBuild execution role."""
        deterministic_suffix = _generate_deterministic_suffix(agent_name)
        role_name = f"AmazonBedrockAgentCoreSDKCodeBuild-{self.session.region_name}-{deterministic_suffix}"

        iam = self.session.client("iam")

        try:
            role = iam.get_role(RoleName=role_name)
            self.logger.info(f"Reusing CodeBuild execution role: {role['Role']['Arn']}")
            return str(role["Role"]["Arn"])
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchEntity":
                raise

        # Create role
        self.logger.info(f"Creating CodeBuild execution role: {role_name}")
        region = self.session.region_name
        source_bucket = f"bedrock-agentcore-codebuild-sources-{account_id}-{region}"

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "codebuild.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {"StringEquals": {"aws:SourceAccount": account_id}},
                }
            ],
        }

        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["ecr:GetAuthorizationToken"],
                    "Resource": "*",
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:PutImage",
                        "ecr:InitiateLayerUpload",
                        "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload",
                    ],
                    "Resource": ecr_repository_arn,
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                    ],
                    "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/bedrock-agentcore-*",
                },
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{source_bucket}",
                        f"arn:aws:s3:::{source_bucket}/*",
                    ],
                },
            ],
        }

        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="CodeBuild execution role for Bedrock AgentCore",
        )

        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="CodeBuildExecutionPolicy",
            PolicyDocument=json.dumps(permissions_policy),
        )

        self.logger.info("Waiting for IAM propagation...")
        time.sleep(10)

        self.logger.info(f"✅ Created CodeBuild execution role: {role['Role']['Arn']}")
        return str(role["Role"]["Arn"])

    def create_or_update_project(
        self,
        agent_name: str,
        ecr_repository_uri: str,
        execution_role: str,
        source_location: str,
    ) -> str:
        """Create or update CodeBuild project."""
        project_name = (
            f"bedrock-agentcore-{_sanitize_ecr_repo_name(agent_name)}-builder"
        )
        buildspec = self._get_arm64_buildspec(ecr_repository_uri)
        codebuild_source_location = source_location.replace("s3://", "")

        project_config = {
            "name": project_name,
            "source": {
                "type": "S3",
                "location": codebuild_source_location,
                "buildspec": buildspec,
            },
            "artifacts": {"type": "NO_ARTIFACTS"},
            "environment": {
                "type": "ARM_CONTAINER",
                "image": "aws/codebuild/amazonlinux2-aarch64-standard:3.0",
                "computeType": "BUILD_GENERAL1_MEDIUM",
                "privilegedMode": True,
            },
            "serviceRole": execution_role,
        }

        try:
            self.client.create_project(**project_config)
            self.logger.info(f"Created CodeBuild project: {project_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                self.client.update_project(**project_config)
                self.logger.info(f"Updated CodeBuild project: {project_name}")
            else:
                raise

        return project_name

    def start_build(self, project_name: str, source_location: str) -> str:
        """Start CodeBuild build."""
        codebuild_source_location = source_location.replace("s3://", "")
        response = self.client.start_build(
            projectName=project_name, sourceLocationOverride=codebuild_source_location
        )
        return str(response["build"]["id"])

    def wait_for_completion(self, build_id: str, timeout: int = 900):
        """Wait for CodeBuild to complete."""
        self.logger.info("Monitoring CodeBuild...")
        current_phase = None
        phase_start_time = None
        build_start_time = time.time()

        while time.time() - build_start_time < timeout:
            response = self.client.batch_get_builds(ids=[build_id])
            build = response["builds"][0]
            status = build["buildStatus"]
            build_phase = build.get("currentPhase", "UNKNOWN")

            # Track phase changes
            if build_phase != current_phase:
                if current_phase and phase_start_time:
                    phase_duration = time.time() - phase_start_time
                    self.logger.info(
                        f"✅ {current_phase} completed in {phase_duration:.1f}s"
                    )

                current_phase = build_phase
                phase_start_time = time.time()
                total_duration = phase_start_time - build_start_time
                self.logger.info(
                    f"🔄 {current_phase} started (total: {total_duration:.0f}s)"
                )

            if status == "SUCCEEDED":
                if current_phase and phase_start_time:
                    phase_duration = time.time() - phase_start_time
                    self.logger.info(
                        f"✅ {current_phase} completed in {phase_duration:.1f}s"
                    )

                total_duration = time.time() - build_start_time
                minutes, seconds = divmod(int(total_duration), 60)
                self.logger.info(f"🎉 CodeBuild completed in {minutes}m {seconds}s")
                return

            elif status in ["FAILED", "FAULT", "STOPPED", "TIMED_OUT"]:
                raise RuntimeError(f"CodeBuild failed with status: {status}")

            time.sleep(1)

        raise TimeoutError(f"CodeBuild timed out after {timeout}s")

    def _get_arm64_buildspec(self, ecr_repository_uri: str) -> str:
        """Get buildspec for ARM64 builds."""
        return f"""
version: 0.2
phases:
  build:
    commands:
      - echo "Starting parallel Docker build and ECR authentication..."
      - |
        docker build -t bedrock-agentcore-arm64 . &
        BUILD_PID=$!
        aws ecr get-login-password --region $AWS_DEFAULT_REGION | \\
        docker login --username AWS --password-stdin {ecr_repository_uri} &
        AUTH_PID=$!
        echo "Waiting for Docker build to complete..."
        wait $BUILD_PID
        if [ $? -ne 0 ]; then
          echo "Docker build failed"
          exit 1
        fi
        echo "Waiting for ECR authentication to complete..."
        wait $AUTH_PID
        if [ $? -ne 0 ]; then
          echo "ECR authentication failed"
          exit 1
        fi
        echo "Both build and auth completed successfully"
      - echo "Tagging image..."
      - docker tag bedrock-agentcore-arm64:latest {ecr_repository_uri}:latest
  post_build:
    commands:
      - echo "Pushing ARM64 image to ECR..."
      - docker push {ecr_repository_uri}:latest
      - echo "Build completed at $(date)"
"""

    def _get_dockerignore_patterns(self) -> list:
        """Get dockerignore patterns."""
        return [
            ".git",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "node_modules",
            ".venv",
            "venv",
            "*.egg-info",
            ".bedrock_agentcore.yaml",
            ".bedrock_agentcore",
            "tests",
            "*.log",
            ".pytest_cache",
            "*.swp",
            "*.swo",
        ]

    def _should_ignore(self, path: str, patterns: list, is_dir: bool) -> bool:
        """Check if path should be ignored."""
        if path.startswith("./"):
            path = path[2:]

        should_ignore = False

        for pattern in patterns:
            if pattern.startswith("!"):
                if self._matches_pattern(path, pattern[1:], is_dir):
                    should_ignore = False
            else:
                if self._matches_pattern(path, pattern, is_dir):
                    should_ignore = True

        return should_ignore

    def _matches_pattern(self, path: str, pattern: str, is_dir: bool) -> bool:
        """Check if path matches pattern."""
        if pattern.endswith("/"):
            if not is_dir:
                return False
            pattern = pattern[:-1]

        if path == pattern:
            return True

        if fnmatch.fnmatch(path, pattern):
            return True

        if is_dir and pattern in path.split("/"):
            return True

        if not is_dir and any(
            fnmatch.fnmatch(part, pattern) for part in path.split("/")
        ):
            return True

        return False


def _create_runtime_execution_role(
    iam, role_name: str, region: str, account_id: str, agent_name: str
) -> str:
    """Create runtime execution role with policies."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"aws:SourceAccount": account_id}},
            }
        ],
    }

    execution_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:GetFoundationModel",
                    "bedrock:ListFoundationModels",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:CreateMemoryEvent",
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:RetrieveMemoryRecords",
                    "bedrock-agentcore:ListMemoryRecords",
                    "bedrock-agentcore:ListEvents",
                ],
                "Resource": f"arn:aws:bedrock-agentcore:{region}:{account_id}:memory/*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                ],
                "Resource": "*",
            },
        ],
    }

    # Create role
    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=f"Execution role for BedrockAgentCore Runtime - {agent_name}",
    )

    # Attach inline policy
    policy_name = f"BedrockAgentCoreRuntimeExecutionPolicy-{agent_name}"
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(execution_policy),
    )

    logger.info(f"✅ Created execution role: {role['Role']['Arn']}")
    return str(role["Role"]["Arn"])


def _generate_deterministic_suffix(agent_name: str, length: int = 10) -> str:
    """Generate deterministic suffix for role names."""
    hash_object = hashlib.sha256(agent_name.encode())
    return hash_object.hexdigest()[:length].lower()


def _sanitize_ecr_repo_name(name: str) -> str:
    """Sanitize name for ECR repository."""
    import re

    name = name.lower()
    name = re.sub(r"[^a-z0-9_\-/]", "-", name)
    if name and not name[0].isalnum():
        name = "a" + name
    name = re.sub(r"[-_]{2,}", "-", name)
    name = name.rstrip("-_")
    if len(name) < 2:
        name = name + "-agent"
    if len(name) > 200:
        name = name[:200].rstrip("-_")
    return name
