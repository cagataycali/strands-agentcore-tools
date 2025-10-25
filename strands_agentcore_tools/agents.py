"""AgentCore Agents Tool - List and manage agent runtimes.

Operations for discovering and managing deployed AgentCore agents.
"""

import json
from typing import Any, Dict, Optional

from strands import tool


@tool
def agents(
    action: str,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    max_results: int = 100,
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """Manage and discover Bedrock AgentCore agent runtimes.

    Args:
        action: Agent operation to perform:
            - "list": List all agent runtimes in the region
            - "get": Get specific agent runtime details
            - "find_by_name": Find agent by name
        agent_id: Agent runtime ID (required for "get" action)
        agent_name: Agent name to search for (required for "find_by_name")
        max_results: Maximum results for list operation (default: 100)
        region: AWS region (default: us-west-2)

    Returns:
        Dict with status and agent information

    Examples:
        # List all agents
        agents(action="list")

        # Get specific agent
        agents(
            action="get",
            agent_id="my-agent-abc123"
        )

        # Find agent by name
        agents(
            action="find_by_name",
            agent_name="my-research-agent"
        )

        # List with limit
        agents(
            action="list",
            max_results=50
        )
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return {
            "status": "error",
            "content": [{"text": "boto3 required. Install: pip install boto3"}],
        }

    try:
        # Initialize control plane client
        client = boto3.client("bedrock-agentcore-control", region_name=region)

        if action == "list":
            all_agents: list[Dict[str, Any]] = []
            next_token = None

            # Handle pagination
            while True:
                params = {"maxResults": min(max_results - len(all_agents), 100)}
                if next_token:
                    params["nextToken"] = next_token

                response = client.list_agent_runtimes(**params)
                agents = response.get("agentRuntimes", [])
                all_agents.extend(agents)

                # Check if we've reached max_results or no more pages
                if len(all_agents) >= max_results:
                    all_agents = all_agents[:max_results]
                    break

                next_token = response.get("nextToken")
                if not next_token:
                    break

            return {
                "status": "success",
                "content": [
                    {"text": f"**Found {len(all_agents)} agent runtimes:**"},
                    {"text": json.dumps(all_agents, indent=2, default=str)},
                ],
            }

        elif action == "get":
            if not agent_id:
                return {
                    "status": "error",
                    "content": [{"text": "agent_id required for get action"}],
                }

            response = client.get_agent_runtime(agentRuntimeId=agent_id)

            agent_info = {
                "agentRuntimeId": response.get("agentRuntimeId"),
                "agentRuntimeArn": response.get("agentRuntimeArn"),
                "agentRuntimeName": response.get("agentRuntimeName"),
                "status": response.get("status"),
                "roleArn": response.get("roleArn"),
                "createdAt": str(response.get("createdAt")),
                "lastUpdatedAt": str(response.get("lastUpdatedAt")),
            }

            # Add container URI if available
            if "agentRuntimeArtifact" in response:
                container_config = response["agentRuntimeArtifact"].get(
                    "containerConfiguration", {}
                )
                if container_config:
                    agent_info["containerUri"] = container_config.get("containerUri")

            # Add network mode if available
            if "networkConfiguration" in response:
                agent_info["networkMode"] = response["networkConfiguration"].get(
                    "networkMode"
                )

            return {
                "status": "success",
                "content": [
                    {"text": "**Agent Runtime Details:**"},
                    {"text": json.dumps(agent_info, indent=2)},
                ],
            }

        elif action == "find_by_name":
            if not agent_name:
                return {
                    "status": "error",
                    "content": [
                        {"text": "agent_name required for find_by_name action"}
                    ],
                }

            # List all agents and search for the name
            all_agents = []
            next_token = None

            while True:
                params = {"maxResults": 100}
                if next_token:
                    params["nextToken"] = next_token

                response = client.list_agent_runtimes(**params)
                agents = response.get("agentRuntimes", [])
                all_agents.extend(agents)

                next_token = response.get("nextToken")
                if not next_token:
                    break

            # Find matching agent
            matching_agent = None
            for agent in all_agents:
                if agent.get("agentRuntimeName") == agent_name:
                    matching_agent = agent
                    break

            if matching_agent:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"✅ **Found agent: {agent_name}**"},
                        {"text": json.dumps(matching_agent, indent=2, default=str)},
                    ],
                }
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Agent not found: {agent_name}"}],
                }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: list, get, find_by_name"
                    }
                ],
            }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        return {
            "status": "error",
            "content": [
                {"text": f"**AWS Error ({error_code}):** {error_message}"},
                {"text": f"**Action:** {action}"},
                {"text": f"**Region:** {region}"},
            ],
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [
                {"text": f"**Unexpected Error:** {str(e)}"},
                {"text": f"**Action:** {action}"},
            ],
        }
