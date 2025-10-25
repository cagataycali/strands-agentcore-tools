"""AgentCore Invoke Tool - Direct boto3 invocation of AgentCore agents.

Invokes deployed Bedrock AgentCore agents using boto3 data plane client.
"""

import json
import uuid
from typing import Any, Dict, Optional

from strands import tool


@tool
def invoke(
    agent_arn: str,
    payload: str,
    session_id: Optional[str] = None,
    qualifier: str = "DEFAULT",
    user_id: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    region: str = "us-west-2",
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Invoke a deployed Bedrock AgentCore agent runtime.

    Args:
        agent_arn: Agent ARN to invoke (required)
            Format: arn:aws:bedrock-agentcore:region:account:runtime/agent-name-id
        payload: JSON payload to send to the agent (required)
            Can be JSON string or will be converted from dict
            Example: '{"prompt": "Hello, what can you do?"}'
        session_id: Runtime session ID for conversation continuity
            If not provided, generates a new UUID
        qualifier: Endpoint qualifier (default: "DEFAULT")
            Use "DEFAULT" for default endpoint or custom endpoint name
        user_id: Optional user ID for authorization flows
        custom_headers: Optional custom headers as dict
            Example: {"X-Amzn-Bedrock-AgentCore-Runtime-Custom-UserId": "user123"}
        region: AWS region (default: us-west-2)

    Returns:
        Dict with status and response content

    Examples:
        # Basic invocation
        invoke(
            agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
            payload='{"prompt": "What can you help me with?"}'
        )

        # With session continuity
        invoke(
            agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
            payload='{"prompt": "Continue our discussion"}',
            session_id="previous-session-id-123"
        )

        # With custom headers
        invoke(
            agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
            payload='{"prompt": "Hello"}',
            custom_headers={"X-Amzn-Bedrock-AgentCore-Runtime-Custom-Context": "production"}
        )
    """
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
    except ImportError:
        return {
            "status": "error",
            "content": [{"text": "boto3 required. Install: pip install boto3"}],
        }

    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Configure boto3 client with appropriate timeouts
        config = Config(
            read_timeout=900,
            connect_timeout=60,
            retries={"max_attempts": 3},
        )

        # Initialize data plane client
        client = boto3.client("bedrock-agentcore", region_name=region, config=config)

        # Prepare request parameters
        params = {
            "agentRuntimeArn": agent_arn,
            "qualifier": qualifier,
            "runtimeSessionId": session_id,
            "payload": payload if isinstance(payload, str) else json.dumps(payload),
        }

        if user_id:
            params["runtimeUserId"] = user_id

        # Handle custom headers using boto3 event system
        handler_id = None
        if custom_headers:

            def add_custom_headers(request, **kwargs):
                for header_name, header_value in custom_headers.items():
                    request.headers.add_header(header_name, header_value)

            handler_id = client.meta.events.register_first(
                "before-sign.bedrock-agentcore.InvokeAgentRuntime", add_custom_headers
            )

        try:
            # Invoke the agent
            response = client.invoke_agent_runtime(**params)

            # Process response
            events = []
            content_type = response.get("contentType", "")

            if "text/event-stream" in content_type:
                # Streaming response - process SSE events
                for chunk in response.get("response", []):
                    # Decode bytes to string
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8")

                    # Split SSE stream by delimiter to get individual events
                    if isinstance(chunk, str):
                        # Split by "\n\ndata: " to separate events
                        parts = chunk.split("\n\ndata: ")
                        # First part may have "data: " prefix
                        if parts and parts[0].startswith("data: "):
                            parts[0] = parts[0][6:]  # Remove "data: " prefix

                        # Process each event
                        for event_str in parts:
                            if not event_str.strip():
                                continue

                            try:
                                # Parse JSON event
                                event = json.loads(event_str)

                                # Stream to callback handler if available
                                if (
                                    agent
                                    and hasattr(agent, "callback_handler")
                                    and agent.callback_handler
                                ):
                                    if isinstance(event, dict):
                                        # Extract text for display from any event type
                                        text_to_display = None

                                        # Check if this is a wrapped AgentCore event
                                        if "event" in event and isinstance(
                                            event["event"], dict
                                        ):
                                            inner = event["event"]
                                            # Extract text from contentBlockDelta
                                            if "contentBlockDelta" in inner:
                                                text_to_display = (
                                                    inner["contentBlockDelta"]
                                                    .get("delta", {})
                                                    .get("text", "")
                                                )
                                            # Pass the inner event
                                            if text_to_display:
                                                agent.callback_handler(
                                                    data=text_to_display, **inner
                                                )
                                            else:
                                                agent.callback_handler(**inner)
                                        # Check if this is a local agent event with 'data' field
                                        elif "data" in event:
                                            text_to_display = event.get("data", "")
                                            # Copy event and remove only non-serializable object references
                                            filtered = event.copy()
                                            for key in [
                                                "agent",
                                                "event_loop_cycle_trace",
                                                "event_loop_cycle_span",
                                            ]:
                                                filtered.pop(key, None)
                                            agent.callback_handler(**filtered)
                                        else:
                                            # Pass other events as-is
                                            agent.callback_handler(**event)

                                # Collect for response
                                events.append(event)
                            except (json.JSONDecodeError, ValueError):
                                # Skip non-JSON content
                                continue
            else:
                # Non-streaming response
                for event in response.get("response", []):
                    if isinstance(event, bytes):
                        try:
                            events.append(event.decode("utf-8"))
                        except UnicodeDecodeError:
                            events.append(str(event))
                    else:
                        events.append(event)

            # Format response
            response_text = (
                "\n".join(str(e) for e in events) if events else "No response content"
            )

            print("\n")

            return {
                "status": "success",
                "content": [
                    {"text": f"**Agent Response:**\n{response_text}"},
                    {"text": f"**Session ID:** {session_id}"},
                    {"text": f"**Endpoint:** {qualifier}"},
                ],
            }

        finally:
            # Clean up event handler
            if handler_id is not None:
                client.meta.events.unregister(
                    "before-sign.bedrock-agentcore.InvokeAgentRuntime", handler_id
                )

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        return {
            "status": "error",
            "content": [
                {"text": f"**AWS Error ({error_code}):** {error_message}"},
                {"text": f"**Agent ARN:** {agent_arn}"},
                {"text": f"**Session ID:** {session_id}"},
            ],
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [
                {"text": f"**Unexpected Error:** {str(e)}"},
                {"text": f"**Agent ARN:** {agent_arn}"},
            ],
        }
