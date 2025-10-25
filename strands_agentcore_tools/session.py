"""AgentCore Session Tool - Manage runtime sessions.

Stop active runtime sessions to free up resources.
"""

from typing import Any, Dict, Optional

from strands import tool


@tool
def session(
    action: str,
    agent_arn: str,
    session_id: str,
    qualifier: str = "DEFAULT",
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """Manage active Bedrock AgentCore runtime sessions with graceful termination and resource cleanup.

    This tool provides comprehensive session lifecycle management for AgentCore agents, enabling
    you to stop active runtime sessions to free resources, handle cleanup during maintenance,
    and manage session lifecycles in production environments. It gracefully handles session
    termination with proper error handling and status reporting.

    How It Works:
    ------------
    1. **Client Initialization**: Creates bedrock-agentcore data plane client with retry configuration
    2. **Action Routing**: Directs request to appropriate session management method
    3. **Session Termination**: Calls StopRuntimeSession API to gracefully stop the session
    4. **Resource Cleanup**: AgentCore automatically frees compute and memory resources
    5. **Status Verification**: Captures HTTP status code and response details
    6. **Error Handling**: Handles session not found gracefully (may already be stopped)
    7. **Response Formatting**: Returns confirmation with session and agent details

    Session Management Process:
    --------------------------
    1. **Validation**: Ensures agent_arn and session_id are provided
    2. **Client Setup**: Configures boto3 with timeouts and retry logic
    3. **API Call**: Executes StopRuntimeSession on bedrock-agentcore data plane
    4. **Response Processing**: Parses status code and confirmation
    5. **Error Handling**: Gracefully handles "already stopped" scenarios
    6. **Result Return**: Provides clear success/error status with context

    Session Lifecycle:
    -----------------
    **Active Session States:**
    ```
    Created → Active → Processing → Idle → Stopped/Expired
    ```

    **Session Creation:**
    - Automatically created on first invoke() call
    - Session ID returned in invoke() response
    - Tracked in .bedrock_agentcore.yaml (agent_session_id)

    **Session Termination:**
    - Manual: Via session(action="stop", ...)
    - Automatic: After idle_timeout expires (if configured)
    - Automatic: After max_lifetime reached (if configured)
    - On disconnect: When client closes connection

    **Resource Impact:**
    - Active sessions consume compute resources
    - Stopping sessions immediately frees resources
    - Stopped sessions cannot be resumed (create new session)
    - Memory persists (if STM/LTM enabled)

    Common Use Cases:
    ---------------

    ### 1. Resource Cleanup After Testing
    ```python
    # After development testing, free resources
    invoke_result = invoke(
        agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
        payload='{"prompt": "test"}'
    )

    # Extract session ID from response or config
    session_id = "abc-123-def-456"

    # Stop session to free resources
    session(
        action="stop",
        agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
        session_id=session_id
    )
    ```

    ### 2. Forced Session Reset
    ```python
    # Force agent to start fresh conversation
    session(
        action="stop",
        agent_arn=agent_arn,
        session_id=old_session_id
    )

    # New invoke will create new session
    invoke(agent_arn=agent_arn, payload='{"prompt": "new conversation"}')
    ```

    ### 3. Maintenance Window Cleanup
    ```python
    # Before maintenance, stop all active sessions
    # Get session IDs from invoke responses or config
    for session_id in active_sessions:
        session(
            action="stop",
            agent_arn=agent_arn,
            session_id=session_id
        )
    ```

    ### 4. Cost Optimization
    ```python
    # Stop long-running idle sessions to save costs
    import time

    # Track last activity
    last_activity = time.time()
    idle_threshold = 3600  # 1 hour

    if time.time() - last_activity > idle_threshold:
        session(action="stop", agent_arn=agent_arn, session_id=session_id)
        print("✅ Idle session stopped to save costs")
    ```

    ### 5. Error Recovery
    ```python
    # If agent becomes unresponsive, force session restart
    try:
        invoke(agent_arn=agent_arn, payload='{"prompt": "test"}')
    except TimeoutError:
        # Stop stuck session
        session(action="stop", agent_arn=agent_arn, session_id=session_id)
        # Retry with new session
        invoke(agent_arn=agent_arn, payload='{"prompt": "test"}')
    ```

    ### 6. Multi-Endpoint Session Management
    ```python
    # Stop session on production endpoint
    session(
        action="stop",
        agent_arn=agent_arn,
        session_id=session_id,
        qualifier="production"
    )

    # Stop session on staging endpoint
    session(
        action="stop",
        agent_arn=agent_arn,
        session_id=staging_session_id,
        qualifier="staging"
    )
    ```

    ### 7. Batch Session Cleanup
    ```python
    # Clean up multiple sessions at once
    session_ids = ["session-1", "session-2", "session-3"]

    for sid in session_ids:
        result = session(
            action="stop",
            agent_arn=agent_arn,
            session_id=sid
        )
        print(f"Session {sid}: {result['status']}")
    ```

    ### 8. Integration with Launch Tool
    ```python
    # launch tool provides stop_session action
    # This is a convenience wrapper that reads config

    # Option 1: Via launch tool (uses config)
    launch(action="stop_session", agent_name="my-agent")

    # Option 2: Direct session tool (no config needed)
    session(
        action="stop",
        agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent",
        session_id="abc-123"
    )
    ```

    ### 9. Graceful Shutdown Pattern
    ```python
    # Before application shutdown, clean up sessions
    import atexit

    def cleanup_sessions():
        for session_id in active_session_ids:
            try:
                session(
                    action="stop",
                    agent_arn=agent_arn,
                    session_id=session_id
                )
            except Exception as e:
                print(f"Cleanup error: {e}")

    atexit.register(cleanup_sessions)
    ```

    ### 10. Session Lifecycle Monitoring
    ```python
    # Monitor and manage long-running sessions
    session_start_time = time.time()
    max_session_duration = 7200  # 2 hours

    while True:
        # Process work...

        if time.time() - session_start_time > max_session_duration:
            # Stop session after 2 hours
            session(action="stop", agent_arn=agent_arn, session_id=session_id)

            # Start new session
            invoke_result = invoke(agent_arn=agent_arn, payload='...')
            break
    ```

    Integration with Other Tools:
    ----------------------------
    **session() works seamlessly with:**

    ```python
    # 1. After invoke - cleanup test sessions
    invoke_result = invoke(
        agent_arn=agent_arn,
        payload='{"prompt": "test"}'
    )
    # Get session_id from response or config
    session(action="stop", agent_arn=agent_arn, session_id=session_id)

    # 2. With status - verify before stopping
    status_result = status(agent_id=agent_id)
    if "READY" in str(status_result):
        session(action="stop", agent_arn=agent_arn, session_id=session_id)

    # 3. With logs - debug session issues
    logs(agent_id=agent_id, action="search", filter_pattern=session_id)
    # Review session logs
    session(action="stop", agent_arn=agent_arn, session_id=session_id)

    # 4. With launch - convenience wrapper
    # launch tool reads config and stops tracked session
    launch(action="stop_session", agent_name="my-agent")

    # Direct session tool doesn't need config
    session(action="stop", agent_arn=agent_arn, session_id=session_id)

    # 5. After multiple invokes - batch cleanup
    results = []
    for i in range(5):
        result = invoke(agent_arn=agent_arn, payload=f'{{"prompt": "test {i}"}}')
        results.append(result)

    # Extract session IDs and cleanup
    for session_id in session_ids:
        session(action="stop", agent_arn=agent_arn, session_id=session_id)
    ```

    Session ID Sources:
    ------------------
    **Where to get session_id:**

    1. **From invoke() response:**
    ```python
    result = invoke(agent_arn=agent_arn, payload='...')
    # Parse response for session_id
    # Note: AgentCore may include session info in response headers
    ```

    2. **From .bedrock_agentcore.yaml:**
    ```yaml
    agents:
      my-agent:
        bedrock_agentcore:
          agent_session_id: abc-123-def-456  # Tracked by launch tool
    ```

    3. **From CloudWatch logs:**
    ```python
    # Session IDs appear in log streams
    logs(agent_name="my-agent", action="streams")
    # Look for: 2024/10/24/[runtime-logs]session-abc123
    ```

    4. **Generate for first invoke:**
    ```python
    import uuid

    session_id = str(uuid.uuid4())
    invoke(agent_arn=agent_arn, payload='...', session_id=session_id)
    ```

    Response Codes:
    --------------
    **Success Response:**
    ```json
    {
      "status": "success",
      "content": [
        {"text": "✅ **Session Stopped Successfully**"},
        {"text": "**Session ID:** abc-123-def-456"},
        {"text": "**Agent ARN:** arn:aws:..."},
        {"text": "**Status Code:** 200"}
      ]
    }
    ```

    **Session Not Found (Graceful):**
    ```json
    {
      "status": "success",
      "content": [
        {"text": "⚠️ **Session Not Found** (may have already been terminated)"},
        {"text": "**Session ID:** abc-123-def-456"}
      ]
    }
    ```

    **Error Response:**
    ```json
    {
      "status": "error",
      "content": [
        {"text": "**AWS Error (AccessDeniedException):** Insufficient permissions"},
        {"text": "**Session ID:** abc-123-def-456"},
        {"text": "**Agent ARN:** arn:aws:..."}
      ]
    }
    ```

    Session Best Practices:
    ----------------------
    **1. Resource Management:**
    - Stop sessions after testing to free resources
    - Don't keep idle sessions active
    - Use idle_timeout in configure for automatic cleanup

    **2. Cost Optimization:**
    - Active sessions consume compute resources
    - Stop long-running idle sessions
    - Configure max_lifetime for automatic termination

    **3. Memory Preservation:**
    - Stopping session doesn't delete memory
    - STM/LTM data persists across sessions
    - New sessions can access historical memory

    **4. Error Handling:**
    - Always handle "session not found" gracefully
    - Session may have already expired
    - Don't fail application if session stop fails

    **5. Session Tracking:**
    - Track active session IDs in application state
    - Use .bedrock_agentcore.yaml for persistence
    - Log session creation/termination events

    Args:
        action: Session operation to perform:
            - "stop": Stop an active runtime session
            Currently only "stop" action is supported
            Future actions: "list", "get", "pause", "resume"

        agent_arn: Agent ARN (required)
            Format: arn:aws:bedrock-agentcore:region:account:runtime/agent-name-id
            Example: "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc123"
            Get from: launch() output, status() response, or .bedrock_agentcore.yaml
            Full ARN required (not just agent ID)

        session_id: Runtime session ID to manage (required)
            Format: UUID string (e.g., "abc-123-def-456")
            Obtained from: invoke() calls, logs, or .bedrock_agentcore.yaml
            Must be an active session for the specified agent
            Cannot reuse stopped session IDs

        qualifier: Endpoint qualifier (default: "DEFAULT")
            Default endpoint: "DEFAULT"
            Custom endpoints: Use custom endpoint name
            Must match endpoint used in invoke()
            Multi-environment: "dev", "staging", "production"

        region: AWS region where agent is deployed (default: us-west-2)
            Must match agent deployment region
            Cross-region session management not supported
            Common regions: us-west-2, us-east-1, eu-west-1

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [
                {"text": "✅ **Session Stopped Successfully**"},
                {"text": "**Session ID:** abc-123-def-456"},
                {"text": "**Agent ARN:** arn:aws:..."},
                {"text": "**Status Code:** 200"}
            ]
        }

        Success case: Returns confirmation with session details and HTTP status
        Error case: Returns AWS error details with session and agent context
        Not found: Returns warning (session may have already terminated)

    Examples:
    --------
    # Basic session stop
    session(
        action="stop",
        agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc123",
        session_id="abc-123-def-456"
    )

    # Stop session on custom endpoint
    session(
        action="stop",
        agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc123",
        session_id="abc-123-def-456",
        qualifier="production"
    )

    # Cleanup after testing
    test_session_id = str(uuid.uuid4())
    invoke(agent_arn=agent_arn, payload='{"prompt": "test"}', session_id=test_session_id)
    # ... run tests ...
    session(action="stop", agent_arn=agent_arn, session_id=test_session_id)

    # Multi-region session management
    session(
        action="stop",
        agent_arn="arn:aws:bedrock-agentcore:us-east-1:123:runtime/global-agent",
        session_id=session_id,
        region="us-east-1"
    )

    # Error recovery - force restart
    try:
        invoke(agent_arn=agent_arn, session_id=stuck_session_id, payload='...')
    except TimeoutError:
        session(action="stop", agent_arn=agent_arn, session_id=stuck_session_id)
        # Retry with new session

    # Batch cleanup
    for session_id in [session1, session2, session3]:
        session(action="stop", agent_arn=agent_arn, session_id=session_id)

    # Graceful shutdown
    def cleanup():
        session(action="stop", agent_arn=agent_arn, session_id=current_session_id)

    atexit.register(cleanup)

    # With launch tool integration
    launch(action="stop_session", agent_name="my-agent")  # Reads config

    # Direct session tool (no config needed)
    session(action="stop", agent_arn=agent_arn, session_id=session_id)

    # Session lifecycle management
    session_age = time.time() - session_start_time
    if session_age > 7200:  # 2 hours
        session(action="stop", agent_arn=agent_arn, session_id=session_id)

    # Debug session via logs before stopping
    logs(agent_id=agent_id, log_stream_name=f"[runtime-logs]{session_id}")
    session(action="stop", agent_arn=agent_arn, session_id=session_id)

    Notes:
        - **Prerequisites**: Agent must be deployed and session must be active
        - **Resource Cleanup**: Stopping session immediately frees compute resources
        - **Memory Persistence**: STM/LTM data persists after session stop
        - **Session Reuse**: Cannot reuse stopped session IDs (create new)
        - **Graceful Handling**: Tool handles "already stopped" scenarios without error
        - **No Resume**: Stopped sessions cannot be resumed (start new session)
        - **Permissions**: Requires `bedrock-agentcore:StopRuntimeSession` permission
        - **Rate Limits**: AWS API rate limits apply (typically sufficient)
        - **Cost**: No direct cost - uses AWS SDK API calls
        - **Best Practice**: Always stop test sessions after use
        - **Timeout Config**: Uses 60s read timeout, 30s connect timeout, 3 retries
        - **HTTP Status**: 200 = success, 404 = not found, 403 = no permission
        - **Idempotent**: Safe to call multiple times (already stopped returns success)
        - **No Side Effects**: Only stops specified session, doesn't affect other sessions
        - **Memory Safe**: Stopping session doesn't delete memory data
        - **Automatic Cleanup**: idle_timeout and max_lifetime auto-stop sessions
        - **Session Tracking**: track session IDs in .bedrock_agentcore.yaml
        - **Multi-Environment**: Each endpoint has independent sessions
        - **Debugging**: Check CloudWatch logs before stopping for troubleshooting
        - **Integration**: Works with launch, invoke, status, and logs tools
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
        if action == "stop":
            # Configure boto3 client with retry logic
            config = Config(
                retries={"max_attempts": 3, "mode": "standard"},
                read_timeout=60,
                connect_timeout=30,
            )

            # Create data plane client
            client = boto3.client(
                "bedrock-agentcore", region_name=region, config=config
            )

            # Stop the runtime session
            response = client.stop_runtime_session(
                agentRuntimeArn=agent_arn,
                runtimeSessionId=session_id,
                qualifier=qualifier,
            )

            # Get HTTP status code
            status_code = response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode", "N/A"
            )

            return {
                "status": "success",
                "content": [
                    {"text": "✅ **Session Stopped Successfully**"},
                    {"text": f"**Session ID:** {session_id}"},
                    {"text": f"**Agent ARN:** {agent_arn}"},
                    {"text": f"**Status Code:** {status_code}"},
                ],
            }

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}. Valid actions: stop"}],
            }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        # Handle session not found gracefully
        if error_code in ["ResourceNotFoundException", "NotFound"]:
            return {
                "status": "success",
                "content": [
                    {
                        "text": "⚠️ **Session Not Found** (may have already been terminated)"
                    },
                    {"text": f"**Session ID:** {session_id}"},
                ],
            }

        return {
            "status": "error",
            "content": [
                {"text": f"**AWS Error ({error_code}):** {error_message}"},
                {"text": f"**Session ID:** {session_id}"},
                {"text": f"**Agent ARN:** {agent_arn}"},
            ],
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [
                {"text": f"**Unexpected Error:** {str(e)}"},
                {"text": f"**Session ID:** {session_id}"},
            ],
        }
