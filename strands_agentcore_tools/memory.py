"""AgentCore Memory Tool - Manage AgentCore Memory resources.

Comprehensive memory operations including create, list, retrieve, and delete.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from strands import tool

# Get logger for this module
logger = logging.getLogger(__name__)

# Configure console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False  # Don't propagate to root logger


@tool
def memory(
    action: str,
    memory_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    event_expiry_days: int = 30,
    strategies: Optional[List[Dict[str, Any]]] = None,
    memory_execution_role_arn: Optional[str] = None,
    encryption_key_arn: Optional[str] = None,
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    search_query: Optional[str] = None,
    top_k: int = 5,
    event_payload: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None,
    max_results: int = 100,
    wait_for_active: bool = False,
    max_wait: int = 300,
    region: str = "us-west-2",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Manage Bedrock AgentCore Memory resources.

    Args:
        action: Memory operation to perform:
            - "create": Create new memory resource
            - "get": Get memory details
            - "list": List all memories
            - "delete": Delete memory resource
            - "create_event": Create memory event for conversation tracking
            - "retrieve": Retrieve memories using semantic search
            - "list_actors": List actors in a memory
            - "list_sessions": List sessions for an actor
            - "get_status": Get memory provisioning status

        memory_id: Memory resource ID (required for most operations)
        name: Memory name (required for create)
        description: Optional description for memory
        event_expiry_days: Event retention in days (default: 30)
        strategies: List of memory strategies for create operation
            Example: [{"semanticMemoryStrategy": {"name": "Facts"}}]
        memory_execution_role_arn: IAM role for memory execution
        encryption_key_arn: KMS key ARN for encryption
        actor_id: Actor ID for event operations
        session_id: Session ID for event operations
        namespace: Namespace for retrieve operations
        search_query: Search text for retrieve operations
        top_k: Number of results for retrieve (default: 5)
        event_payload: Event payload for create_event operation
        max_results: Maximum results for list operations (default: 100)
        wait_for_active: Wait for memory to become ACTIVE after create (default: False)
        max_wait: Maximum wait time in seconds (default: 300)
        region: AWS region (default: us-west-2)
        verbose: Enable verbose logging with detailed progress (default: False)

    Returns:
        Dict with status and operation results

    Examples:
        # Create STM-only memory
        memory(
            action="create",
            name="my-agent-memory",
            description="Memory for my agent"
        )

        # Create memory with LTM strategies
        memory(
            action="create",
            name="my-agent-memory",
            strategies=[
                {"semanticMemoryStrategy": {"name": "Facts", "namespaces": ["/users/{actorId}/facts"]}},
                {"userPreferenceMemoryStrategy": {"name": "Preferences", "namespaces": ["/users/{actorId}/prefs"]}},
                {"summarizationMemoryStrategy": {"name": "Summaries", "namespaces": ["/summaries/{actorId}"]}}
            ],
            wait_for_active=True
        )

        # List all memories
        memory(action="list")

        # Get memory details
        memory(action="get", memory_id="my-memory-abc123")

        # Create conversation event
        memory(
            action="create_event",
            memory_id="my-memory-abc123",
            actor_id="user-123",
            session_id="session-456",
            event_payload={"conversational": {"content": "Hello!", "role": "user"}}
        )

        # Retrieve memories semantically
        memory(
            action="retrieve",
            memory_id="my-memory-abc123",
            namespace="/users/user-123/facts",
            search_query="What are user preferences?",
            top_k=5
        )

        # Delete memory
        memory(action="delete", memory_id="my-memory-abc123")
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return {
            "status": "error",
            "content": [{"text": "boto3 required. Install: pip install boto3"}],
        }

    # Print verbose messages directly for visibility
    if verbose:
        print(f"🔧 Starting memory operation: {action}", flush=True)

    try:
        # Initialize both clients
        control_client = boto3.client("bedrock-agentcore-control", region_name=region)
        data_client = boto3.client("bedrock-agentcore", region_name=region)

        if verbose:
            print(f"✅ Initialized clients for region: {region}", flush=True)

        # Route to appropriate operation
        if action == "create":
            if not name:
                return {
                    "status": "error",
                    "content": [{"text": "name is required for create action"}],
                }

            if verbose:
                print(f"📝 Creating memory: {name}", flush=True)

            params = {
                "name": name,
                "eventExpiryDuration": event_expiry_days,
                "clientToken": str(uuid.uuid4()),
            }

            if description:
                params["description"] = description
            if strategies:
                params["memoryStrategies"] = strategies
                if verbose:
                    print(
                        f"🧠 Memory strategies: {len(strategies)} configured",
                        flush=True,
                    )
            if memory_execution_role_arn:
                params["memoryExecutionRoleArn"] = memory_execution_role_arn
            if encryption_key_arn:
                params["encryptionKeyArn"] = encryption_key_arn

            if verbose:
                print(f"🚀 Calling CreateMemory API...", flush=True)

            response = control_client.create_memory(**params)
            memory = response["memory"]

            memory_id = memory.get("id", memory.get("memoryId"))
            status = memory.get("status")

            if verbose:
                print(
                    f"✅ Memory created with ID: {memory_id}, Status: {status}",
                    flush=True,
                )

            result_content = [
                {"text": "✅ **Memory Created Successfully**"},
                {"text": f"**Memory ID:** {memory_id}"},
                {"text": f"**Status:** {status}"},
                {"text": f"**Region:** {region}"},
            ]

            # If wait_for_active is True, poll until ACTIVE
            if wait_for_active and status != "ACTIVE":
                result_content.append(
                    {"text": "\n⏳ **Waiting for memory to become ACTIVE...**"}
                )

                if verbose:
                    print(
                        f"⏳ Polling for ACTIVE status (max {max_wait}s)...", flush=True
                    )

                start_time = time.time()
                while time.time() - start_time < max_wait:
                    elapsed = int(time.time() - start_time)

                    try:
                        status_response = control_client.get_memory(memoryId=memory_id)
                        current_status = status_response["memory"]["status"]

                        if verbose:
                            print(
                                f"📊 Status check {elapsed}s: {current_status}",
                                flush=True,
                            )

                        if current_status == "ACTIVE":
                            result_content.append(
                                {"text": f"\n✅ **Memory is ACTIVE** (took {elapsed}s)"}
                            )
                            if verbose:
                                print(
                                    f"🎉 Memory is ACTIVE after {elapsed}s!", flush=True
                                )
                            break
                        elif current_status == "FAILED":
                            failure_reason = status_response["memory"].get(
                                "failureReason", "Unknown"
                            )
                            if verbose:
                                print(f"❌ Memory creation failed: {failure_reason}")
                            return {
                                "status": "error",
                                "content": [
                                    {
                                        "text": f"❌ **Memory creation failed:** {failure_reason}"
                                    }
                                ],
                            }

                        time.sleep(10)
                    except ClientError as e:
                        if verbose:
                            print(f"❌ Error checking status: {e}", flush=True)
                        return {
                            "status": "error",
                            "content": [{"text": f"Error checking status: {str(e)}"}],
                        }
                else:
                    if verbose:
                        print(f"⚠️ Timeout after {max_wait}s - still provisioning")
                    result_content.append(
                        {
                            "text": f"\n⚠️ **Timeout** after {max_wait}s - memory still provisioning"
                        }
                    )

            return {"status": "success", "content": result_content}

        elif action == "get":
            if not memory_id:
                return {
                    "status": "error",
                    "content": [{"text": "memory_id is required for get action"}],
                }

            if verbose:
                print(f"🔍 Getting memory details: {memory_id}", flush=True)

            response = control_client.get_memory(memoryId=memory_id)
            memory = response["memory"]

            if verbose:
                print(f"✅ Retrieved memory: {memory.get('name')}", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": "**Memory Details:**"},
                    {"text": json.dumps(memory, indent=2, default=str)},
                ],
            }

        elif action == "list":
            if verbose:
                print(f"📋 Listing memories (max {max_results})...", flush=True)

            response = control_client.list_memories(maxResults=min(max_results, 100))
            memories = response.get("memories", [])

            # Handle pagination
            next_token = response.get("nextToken")
            page = 1
            while next_token and len(memories) < max_results:
                page += 1
                remaining = max_results - len(memories)

                if verbose:
                    print(f"📄 Fetching page {page} (total: {len(memories)} so far)...")

                response = control_client.list_memories(
                    maxResults=min(remaining, 100), nextToken=next_token
                )
                memories.extend(response.get("memories", []))
                next_token = response.get("nextToken")

            if verbose:
                print(f"✅ Found {len(memories)} memories total", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"**Found {len(memories)} memories:**"},
                    {"text": json.dumps(memories, indent=2, default=str)},
                ],
            }

        elif action == "delete":
            if not memory_id:
                return {
                    "status": "error",
                    "content": [{"text": "memory_id is required for delete action"}],
                }

            if verbose:
                print(f"🗑️ Deleting memory: {memory_id}", flush=True)

            control_client.delete_memory(
                memoryId=memory_id, clientToken=str(uuid.uuid4())
            )

            if verbose:
                print(f"✅ Memory deleted successfully", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": "✅ **Memory Deleted Successfully**"},
                    {"text": f"**Memory ID:** {memory_id}"},
                ],
            }

        elif action == "create_event":
            if not memory_id or not actor_id or not event_payload:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "memory_id, actor_id, and event_payload required for create_event"
                        }
                    ],
                }

            if verbose:
                print(f"💬 Creating event for actor: {actor_id}, session: {session_id}")

            # Normalize event_payload format
            if isinstance(event_payload, list):
                payload = event_payload
            else:
                payload = [event_payload]

            # Ensure conversational payloads have correct structure
            normalized_payload = []
            for item in payload:
                if item and isinstance(item, dict) and "conversational" in item:
                    conv = item["conversational"]
                    if isinstance(conv, dict):
                        # Ensure content is a dict with 'text' key
                        if isinstance(conv.get("content"), str):
                            conv["content"] = {"text": conv["content"]}
                        # Ensure role is uppercase
                        if "role" in conv:
                            conv["role"] = conv["role"].upper()
                normalized_payload.append(item)

            params = {
                "memoryId": memory_id,
                "actorId": actor_id,
                "eventTimestamp": (
                    datetime.now(datetime.UTC)
                    if hasattr(datetime, "UTC")
                    else datetime.utcnow()
                ),
                "payload": normalized_payload,
            }

            if session_id:
                params["sessionId"] = session_id

            if verbose:
                print(f"🚀 Calling CreateEvent API...", flush=True)

            response = data_client.create_event(**params)

            if verbose:
                print(f"✅ Event created successfully", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": "✅ **Event Created Successfully**"},
                    {"text": f"**Event ID:** {response.get('eventId')}"},
                    {"text": f"**Memory ID:** {memory_id}"},
                ],
            }

        elif action == "retrieve":
            if not all([memory_id, namespace, search_query]):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "memory_id, namespace, and search_query required for retrieve"
                        }
                    ],
                }

            if verbose:
                print(f"🔎 Retrieving memories from namespace: {namespace}", flush=True)
                print(f"🔎 Search query: {search_query}, top_k: {top_k}", flush=True)

            params = {
                "memoryId": memory_id,
                "namespace": namespace,
                "searchCriteria": {"searchQuery": search_query, "topK": top_k},
            }

            response = data_client.retrieve_memory_records(**params)
            records = response.get("memoryRecords", [])

            if verbose:
                print(f"✅ Retrieved {len(records)} memory records", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"**Retrieved {len(records)} memory records:**"},
                    {"text": json.dumps(records, indent=2, default=str)},
                ],
            }

        elif action == "list_actors":
            if not memory_id:
                return {
                    "status": "error",
                    "content": [{"text": "memory_id required for list_actors"}],
                }

            if verbose:
                print(f"👥 Listing actors for memory: {memory_id}", flush=True)

            params = {"memoryId": memory_id, "maxResults": min(max_results, 100)}

            response = data_client.list_actors(**params)
            actors = response.get("actors", [])

            if verbose:
                print(f"✅ Found {len(actors)} actors", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"**Found {len(actors)} actors:**"},
                    {"text": json.dumps(actors, indent=2, default=str)},
                ],
            }

        elif action == "list_sessions":
            if not all([memory_id, actor_id]):
                return {
                    "status": "error",
                    "content": [
                        {"text": "memory_id and actor_id required for list_sessions"}
                    ],
                }

            if verbose:
                print(f"📱 Listing sessions for actor: {actor_id}", flush=True)

            params = {
                "memoryId": memory_id,
                "actorId": actor_id,
                "maxResults": min(max_results, 100),
            }

            response = data_client.list_sessions(**params)
            sessions = response.get("sessions", [])

            if verbose:
                print(f"✅ Found {len(sessions)} sessions", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"**Found {len(sessions)} sessions:**"},
                    {"text": json.dumps(sessions, indent=2, default=str)},
                ],
            }

        elif action == "get_status":
            if not memory_id:
                return {
                    "status": "error",
                    "content": [{"text": "memory_id required for get_status"}],
                }

            if verbose:
                print(f"📊 Checking status for memory: {memory_id}", flush=True)

            response = control_client.get_memory(memoryId=memory_id)
            status = response["memory"]["status"]

            if verbose:
                print(f"✅ Memory status: {status}", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"**Memory Status:** {status}"},
                    {"text": f"**Memory ID:** {memory_id}"},
                ],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: create, get, list, delete, "
                        "create_event, retrieve, list_actors, list_sessions, get_status"
                    }
                ],
            }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        if verbose:
            print(f"❌ AWS Error: {error_code} - {error_message}", flush=True)

        return {
            "status": "error",
            "content": [
                {"text": f"**AWS Error ({error_code}):** {error_message}"},
                {"text": f"**Action:** {action}"},
                {"text": f"**Region:** {region}"},
            ],
        }

    except Exception as e:
        if verbose:
            print(f"❌ Unexpected Error: {str(e)}", flush=True)

        return {
            "status": "error",
            "content": [
                {"text": f"**Unexpected Error:** {str(e)}"},
                {"text": f"**Action:** {action}"},
            ],
        }
