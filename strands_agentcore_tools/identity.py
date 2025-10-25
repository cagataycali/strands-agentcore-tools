"""AgentCore Identity Tool - Manage OAuth2, API Keys, and Workload Identities.

Comprehensive identity management for AgentCore agents.
"""

import json
from typing import Any, Dict, List, Optional

from strands import tool


@tool
def identity(
    action: str,
    name: Optional[str] = None,
    provider_type: str = "oauth2",
    # OAuth2 specific
    vendor: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    discovery_url: Optional[str] = None,
    authorization_endpoint: Optional[str] = None,
    token_endpoint: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    # API Key specific
    api_key: Optional[str] = None,
    header_name: Optional[str] = None,
    # Workload Identity specific
    workload_arn: Optional[str] = None,
    # Token Vault
    kms_key_id: Optional[str] = None,
    # Common
    max_results: int = 20,
    region: str = "us-west-2",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Manage AgentCore Identity - OAuth2, API keys, secure credential storage for agents.

    How It Works:
    ------------
    1. **Provider Creation**: Register OAuth2 or API key providers with AgentCore
    2. **Secure Storage**: Credentials stored in AWS Secrets Manager with KMS encryption
    3. **Agent Integration**: Use @requires_access_token decorator in agent code
    4. **Token Retrieval**: SDK automatically handles OAuth flows and token management
    5. **Agent Invocation**: Agents receive tokens to access external services
    6. **Audit Trail**: All operations logged for compliance and security monitoring

    Identity Integration Patterns:
    -----------------------------

    ### Pattern 1: OAuth2 User-Delegated Access (@requires_access_token)
    Build agents that access user data with explicit consent:

    ```python
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
    from bedrock_agentcore.identity import requires_access_token
    import asyncio

    app = BedrockAgentCoreApp()
    queue = asyncio.Queue()

    async def handle_auth_url(url):
        await queue.put(f"Authorize: {url}")

    @requires_access_token(
        provider_name="my-slack-oauth",
        scopes=["users:read", "channels:read"],
        auth_flow="USER_FEDERATION",
        on_auth_url=handle_auth_url,
        force_authentication=True
    )
    async def access_slack_data(*, access_token: str):
        # Token automatically provided by decorator
        headers = {"Authorization": f"Bearer {access_token}"}
        # Make authenticated API calls...
        await queue.put({"message": "Accessed user's Slack!"})

    @app.entrypoint
    async def invoke(payload, context):
        task = asyncio.create_task(access_slack_data())
        async for item in queue.stream():
            yield item
        await task
    ```

    ### Pattern 2: API Key Machine-to-Machine Access
    Build agents that use API keys for service authentication:

    ```python
    # 1. Store API key in AgentCore Identity
    identity(
        action="create",
        provider_type="api_key",
        name="github-api",
        api_key="ghp_xxxxx"
    )

    # 2. Retrieve in agent code
    import boto3
    client = boto3.client("bedrock-agentcore-control")
    response = client.get_api_key_credential_provider(name="github-api")

    # 3. Get secret from Secrets Manager
    secret_arn = response["apiKeySecretArn"]["secretArn"]
    secrets = boto3.client("secretsmanager")
    api_key = secrets.get_secret_value(SecretId=secret_arn)["SecretString"]
    ```

    ### Pattern 3: Workload Identity (Agent-to-Agent)
    Secure communication between deployed agents:

    ```python
    # Automatically created when agent is deployed
    # Each agent gets workload identity:
    # arn:aws:bedrock-agentcore:region:account:workload-identity-directory/default/workload-identity/agent-name

    # Use in IAM policies for agent-to-agent authorization
    ```

    Building Identity-Connected Agents:
    -----------------------------------

    ### Complete Agent Example with OAuth2:

    ```python
    \"\"\"
    Slack Integration Agent - Reads user's channels with OAuth2
    \"\"\"
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
    from bedrock_agentcore.identity import requires_access_token
    import asyncio
    import httpx
    import logging

    app = BedrockAgentCoreApp()
    logger = logging.getLogger(__name__)

    class StreamingQueue:
        def __init__(self):
            self.finished = False
            self.queue = asyncio.Queue()

        async def put(self, item):
            await self.queue.put(item)

        async def finish(self):
            self.finished = True
            await self.queue.put(None)

        async def stream(self):
            while True:
                item = await self.queue.get()
                if item is None and self.finished:
                    break
                yield item

    queue = StreamingQueue()

    async def handle_auth_url(url):
        # Stream authorization URL to user
        await queue.put(f"Please authorize: {url}")

    @requires_access_token(
        provider_name="my-slack-oauth",
        scopes=["channels:read", "users:read"],
        auth_flow="USER_FEDERATION",
        on_auth_url=handle_auth_url,
        force_authentication=False  # Reuse cached tokens
    )
    async def list_slack_channels(*, access_token: str):
        logger.info("Got access token, fetching channels...")

        # Use token to access Slack API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://slack.com/api/conversations.list",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            data = response.json()

            if data.get("ok"):
                channels = [ch["name"] for ch in data.get("channels", [])]
                await queue.put({
                    "message": "Successfully accessed Slack!",
                    "channels": channels
                })
            else:
                await queue.put({"error": data.get("error")})

        await queue.finish()

    @app.entrypoint
    async def invoke(payload, context):
        logger.info("Starting Slack agent...")

        # Start OAuth flow and channel retrieval
        task = asyncio.create_task(list_slack_channels())

        # Stream results
        async for item in queue.stream():
            yield item

        await task

    if __name__ == "__main__":
        app.run()
    ```

    Deployment Workflow:
    ----------------------------

    ### Step 1: Create Credential Provider (This Tool)

    ```python
    # Create OAuth2 provider for Slack
    identity(
        action="create",
        provider_type="oauth2",
        name="my-slack-oauth",
        vendor="SlackOauth2",
        client_id="8911857480961.9155488574208",
        client_secret="your-client-secret"
    )

    # Create API key provider for GitHub
    identity(
        action="create",
        provider_type="api_key",
        name="github-api",
        api_key="ghp_xxxxx"
    )
    ```

    ### Step 2: Write Agent Code (See examples above)

    ### Step 3: Configure Agent with Identity Permissions

    ```python
    # Configure agent
    configure(
        action="configure",
        entrypoint="slack_agent.py",
        agent_name="slack-integration",
        memory_mode="STM_AND_LTM"
    )

    # Add IAM permissions for identity access
    # The agent needs permissions to:
    # - bedrock-agentcore:GetResourceOauth2Token
    # - secretsmanager:GetSecretValue
    ```

    ### Step 4: Deploy Agent

    ```python
    # Launch to AgentCore
    launch(
        action="launch",
        agent_name="slack-integration",
        mode="codebuild"
    )
    ```

    ### Step 5: Invoke with User Context

    ```python
    # Invoke agent (triggers OAuth flow)
    invoke(
        agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/slack-integration",
        payload='{"prompt": "List my Slack channels"}',
        user_id="user-123",  # Required for OAuth flows
        session_id="33-char-minimum-session-id-here"  # Min 33 chars
    )

    # Agent returns authorization URL
    # User visits URL → Grants consent → Agent receives token
    # Subsequent calls reuse cached token (no re-auth)
    ```

    IAM Permissions Required:
    -------------------------

    **For Agent Execution Role:**

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "AccessTokenVault",
          "Effect": "Allow",
          "Action": [
            "bedrock-agentcore:GetResourceOauth2Token",
            "secretsmanager:GetSecretValue"
          ],
          "Resource": [
            "arn:aws:bedrock-agentcore:region:account:workload-identity-directory/default/workload-identity/*",
            "arn:aws:bedrock-agentcore:region:account:token-vault/default/oauth2credentialprovider/*",
            "arn:aws:secretsmanager:region:account:secret:bedrock-agentcore-identity!default/oauth2/*"
          ]
        }
      ]
    }
    ```

    **For This Tool (Management Operations):**

    ```json
    {
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore-control:CreateOAuth2CredentialProvider",
        "bedrock-agentcore-control:GetOAuth2CredentialProvider",
        "bedrock-agentcore-control:ListOAuth2CredentialProviders",
        "bedrock-agentcore-control:DeleteOAuth2CredentialProvider",
        "bedrock-agentcore-control:CreateApiKeyCredentialProvider",
        "bedrock-agentcore-control:GetApiKeyCredentialProvider",
        "bedrock-agentcore-control:ListApiKeyCredentialProviders",
        "bedrock-agentcore-control:DeleteApiKeyCredentialProvider"
      ],
      "Resource": "*"
    }
    ```

    Common Use Cases:
    ----------------

    ### 1. Slack Bot Integration

    ```python
    # Store Slack OAuth credentials
    identity(
        action="create",
        provider_type="oauth2",
        name="slack-bot",
        vendor="SlackOauth2",
        client_id="8911857480961.9644037846534",
        client_secret="6068b88027530422b4122490826d1363"
    )

    # Agent uses @requires_access_token to access Slack API
    # Supports: message posting, channel management, user lookup
    ```

    ### 2. GitHub Repository Access

    ```python
    # Store GitHub API token
    identity(
        action="create",
        provider_type="api_key",
        name="github-token",
        api_key="ghp_xxxxx"
    )

    # Agent retrieves token from Secrets Manager
    # Supports: repo access, issue management, PR operations
    ```

    ### 3. Multi-Service Integration

    ```python
    # Create multiple providers for comprehensive integration
    providers = [
        ("slack-oauth", "SlackOauth2", slack_creds),
        ("github-oauth", "GitHubOauth2", github_creds),
        ("google-oauth", "GoogleOauth2", google_creds)
    ]

    for name, vendor, creds in providers:
        identity(
            action="create",
            provider_type="oauth2",
            name=name,
            vendor=vendor,
            client_id=creds["id"],
            client_secret=creds["secret"]
        )

    # Agent uses multiple @requires_access_token decorators
    # Each for different service integration
    ```

    ### 4. API Key Management

    ```python
    # Centralize all API keys in AgentCore Identity
    api_keys = {
        "brave-search": "BSAUQHelAzCuBPgZhmhmzF1V0v5l0Ex",
        "huggingface": "hf_xxxxx",
        "shodan": "9VJCqKL5eKn7vFwuCZI7j8R5aOLVUIv6"
    }

    for name, key in api_keys.items():
        identity(
            action="create",
            provider_type="api_key",
            name=name,
            api_key=key
        )

    # Benefits:
    # - No hardcoded secrets in code
    # - Centralized rotation
    # - Audit trail for all access
    ```

    ### 5. Workload Identity Verification

    ```python
    # List all agent workload identities
    identity(action="list", provider_type="workload")

    # Verify specific agent identity
    identity(
        action="get",
        provider_type="workload",
        name="my-agent-name"
    )

    # Use for agent-to-agent authorization policies
    ```

    OAuth2 Flow Types:
    -----------------

    ### 2LO (Client Credentials Grant) - Machine-to-Machine

    **Use when:** Agent needs to access resources as itself (no user context)

    **Example:**
    ```python
    @requires_access_token(
        provider_name="service-oauth",
        scopes=["api.read"],
        auth_flow="CLIENT_CREDENTIALS"  # 2LO flow
    )
    async def access_service(*, access_token: str):
        # Agent authenticates as itself
        # No user interaction needed
        pass
    ```

    ### 3LO (Authorization Code Grant) - User-Delegated Access

    **Use when:** Agent needs to access user-specific data (requires consent)

    **Example:**
    ```python
    @requires_access_token(
        provider_name="slack-oauth",
        scopes=["channels:read"],
        auth_flow="USER_FEDERATION",  # 3LO flow
        on_auth_url=handle_auth_url,  # Stream URL to user
        force_authentication=True  # Always request fresh consent
    )
    async def access_user_slack(*, access_token: str):
        # Agent accesses user's Slack data
        # User must grant consent first
        pass
    ```

    Integration with Other Tools:
    -----------------------------

    **identity() works seamlessly with:**

    ```python
    # 1. With configure - set up identity permissions
    configure(
        action="configure",
        entrypoint="oauth_agent.py",
        agent_name="my-oauth-agent"
    )
    # Then add IAM permissions for identity access

    # 2. With launch - deploy identity-enabled agents
    identity(action="create", provider_type="oauth2", name="slack-oauth", ...)
    launch(action="launch", agent_name="my-oauth-agent")

    # 3. With invoke - trigger OAuth flows
    invoke(
        agent_arn="...",
        payload='{"prompt": "Access my Slack"}',
        user_id="user-123",  # Required for OAuth flows
        session_id="min-33-chars-session-id"
    )

    # 4. With status - verify agent readiness
    status(agent_id="my-oauth-agent-abc")
    invoke(agent_arn="...", user_id="user-123", ...)

    # 5. With logs - debug OAuth flows
    logs(
        agent_name="my-oauth-agent",
        action="search",
        filter_pattern="OAuth"
    )
    ```

    Security Best Practices:
    -----------------------

    **1. Never Hardcode Secrets:**
    - ❌ Don't: `api_key = "ghp_xxxxx"` in code
    - ✅ Do: Store in AgentCore Identity, retrieve at runtime

    **2. Use Least Privilege:**
    - Only request scopes your agent needs
    - Example: `scopes=["channels:read"]` not `scopes=["*"]`

    **3. Rotate Credentials:**
    ```python
    # Update OAuth client secret
    identity(
        action="update",
        provider_type="oauth2",
        name="slack-oauth",
        client_secret="new-secret-here"
    )
    ```

    **4. Audit Access:**
    ```python
    # List all credential providers
    identity(action="list", provider_type="oauth2")
    identity(action="list", provider_type="api_key")

    # Review token vault encryption
    identity(action="get_vault")
    ```

    **5. Use Customer-Managed KMS Keys:**
    ```python
    # Set custom KMS key for token vault
    identity(
        action="set_vault_key",
        kms_key_id="arn:aws:kms:us-west-2:123:key/abc-123"
    )
    ```

    Supported OAuth2 Vendors:
    -------------------------

    | Vendor | Discovery URL | Typical Scopes |
    |--------|--------------|----------------|
    | **SlackOauth2** | https://slack.com/.well-known/openid-configuration | channels:read, users:read, chat:write |
    | **GitHubOauth2** | https://github.com/.well-known/openid-configuration | repo, user, admin:org |
    | **GoogleOauth2** | https://accounts.google.com/.well-known/openid-configuration | openid, profile, email |
    | **CustomOauth2** | Your discovery URL | Custom scopes |

    Args:
        action: Identity operation to perform:
            - "create": Create new credential provider or workload identity
            - "get": Get credential provider or workload identity details
            - "list": List all credential providers or workload identities
            - "delete": Delete credential provider or workload identity
            - "update": Update credential provider
            - "get_vault": Get token vault configuration
            - "set_vault_key": Set token vault KMS key

        name: Identity resource name (required for most operations)
            Format: [a-zA-Z0-9_-]{1,100}
            Example: "my-slack-oauth", "github-api-key"

        provider_type: Type of credential provider:
            - "oauth2": OAuth2 credential provider (default)
              Use for: User-delegated access, service OAuth
            - "api_key": API key credential provider
              Use for: Simple API authentication
            - "workload": Workload identity
              Use for: Agent-to-agent authorization

        # OAuth2 Parameters
        vendor: OAuth2 vendor (required for OAuth2 create):
            - "SlackOauth2": Slack OAuth with auto-discovery
            - "GitHubOauth2": GitHub OAuth with auto-discovery
            - "GoogleOauth2": Google OAuth with auto-discovery
            - "CustomOauth2": Custom OAuth provider (requires discovery_url)

        client_id: OAuth2 client ID from your OAuth app
            Get from: OAuth provider's app settings
            Example: "8911857480961.9155488574208"

        client_secret: OAuth2 client secret from your OAuth app
            Stored in: AWS Secrets Manager (encrypted)
            Never logged or returned in responses

        discovery_url: OAuth2 discovery URL (for CustomOauth2)
            Format: https://provider.com/.well-known/openid-configuration
            Contains: Authorization/token endpoints, supported scopes

        authorization_endpoint: Authorization endpoint URL (for CustomOauth2)
            Example: "https://provider.com/oauth/authorize"

        token_endpoint: Token endpoint URL (for CustomOauth2)
            Example: "https://provider.com/oauth/token"

        scopes: List of OAuth2 scopes (optional)
            Example: ["openid", "profile", "email"]
            Provider-specific scopes defined by OAuth server

        # API Key Parameters
        api_key: API key value (required for API key create)
            Stored in: AWS Secrets Manager (encrypted)
            Use for: Simple API authentication

        header_name: HTTP header name for API key
            Example: "X-API-Key", "Authorization"
            Note: Currently not used by API (stored in app config)

        # Workload Identity Parameters
        workload_arn: Workload ARN to associate with identity
            Format: arn:aws:bedrock-agentcore:region:account:runtime/agent-name
            Auto-created: When agent is deployed to AgentCore

        # Token Vault Parameters
        kms_key_id: KMS key ID for token vault encryption
            Format: arn:aws:kms:region:account:key/key-id
            Default: Service-managed KMS key
            Custom: For compliance requirements

        # Common Parameters
        max_results: Maximum results for list operations (default: 20)
            Range: 1-100
            Use: Pagination control

        region: AWS region where identity resources are stored
            Default: us-west-2
            Must match: Agent deployment region

        verbose: Enable verbose logging (default: False)
            True: Shows detailed progress with emoji indicators
            False: Silent operation
            Output: Goes to stdout/CloudWatch Logs

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [
                {"text": "Operation result message"},
                {"text": "Additional details"}
            ]
        }

        Success case: Returns provider ARN, callback URLs, configuration
        Error case: Returns error details with troubleshooting guidance

    Examples:
        # Create Slack OAuth2 credential provider
        identity(
            action="create",
            provider_type="oauth2",
            name="my-slack-oauth",
            vendor="SlackOauth2",
            client_id="8911857480961.9155488574208",
            client_secret="your-secret-here"
        )

        # Create GitHub OAuth2 credential provider
        identity(
            action="create",
            provider_type="oauth2",
            name="my-github-oauth",
            vendor="GitHubOauth2",
            client_id="github-client-id",
            client_secret="github-client-secret",
            discovery_url="https://github.com/.well-known/openid-configuration"
        )

        # Create generic OAuth2 provider
        identity(
            action="create",
            provider_type="oauth2",
            name="my-oauth-provider",
            vendor="GenericOauth2",
            client_id="client-id",
            client_secret="client-secret",
            authorization_endpoint="https://auth.example.com/oauth/authorize",
            token_endpoint="https://auth.example.com/oauth/token",
            scopes=["read", "write"]
        )

        # Create API key credential provider
        identity(
            action="create",
            provider_type="api_key",
            name="my-api-key",
            api_key="your-api-key-here",
            header_name="X-API-Key"
        )

        # List all OAuth2 providers
        identity(action="list", provider_type="oauth2")

        # List all workload identities
        identity(action="list", provider_type="workload")

        # Get specific OAuth2 provider
        identity(
            action="get",
            provider_type="oauth2",
            name="my-slack-oauth"
        )

        # Delete OAuth2 provider
        identity(
            action="delete",
            provider_type="oauth2",
            name="my-slack-oauth"
        )

        # Get token vault configuration
        identity(action="get_vault")

        # Set token vault KMS key
        identity(
            action="set_vault_key",
            kms_key_id="arn:aws:kms:us-west-2:123:key/abc-123"
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

    if verbose:
        print(f"🔧 Starting identity operation: {action} ({provider_type})", flush=True)

    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)

        if verbose:
            print(f"✅ Initialized client for region: {region}", flush=True)

        # Route to appropriate operation
        if action == "create":
            if not name:
                return {
                    "status": "error",
                    "content": [{"text": "name is required for create action"}],
                }

            if provider_type == "oauth2":
                return _create_oauth2_provider(
                    client,
                    name,
                    vendor,
                    client_id,
                    client_secret,
                    discovery_url,
                    authorization_endpoint,
                    token_endpoint,
                    scopes,
                    verbose,
                )
            elif provider_type == "api_key":
                return _create_api_key_provider(
                    client, name, api_key, header_name, verbose
                )
            elif provider_type == "workload":
                return _create_workload_identity(client, name, workload_arn, verbose)
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Unknown provider_type: {provider_type}. Valid: oauth2, api_key, workload"
                        }
                    ],
                }

        elif action == "get":
            if not name:
                return {
                    "status": "error",
                    "content": [{"text": "name is required for get action"}],
                }

            if verbose:
                print(f"🔍 Getting {provider_type} provider: {name}", flush=True)

            if provider_type == "oauth2":
                response = client.get_oauth2_credential_provider(name=name)
                result = response
            elif provider_type == "api_key":
                response = client.get_api_key_credential_provider(name=name)
                result = response
            elif provider_type == "workload":
                response = client.get_workload_identity(name=name)
                result = response
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Unknown provider_type: {provider_type}. Valid: oauth2, api_key, workload"
                        }
                    ],
                }

            if verbose:
                print(f"✅ Retrieved {provider_type} provider", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"**{provider_type.upper()} Provider Details:**"},
                    {"text": json.dumps(result, indent=2, default=str)},
                ],
            }

        elif action == "list":
            if verbose:
                print(
                    f"📋 Listing {provider_type} providers (max {max_results})...",
                    flush=True,
                )

            if provider_type == "oauth2":
                response = client.list_oauth2_credential_providers(
                    maxResults=min(max_results, 100)
                )
                items = response.get("credentialProviders", [])
            elif provider_type == "api_key":
                response = client.list_api_key_credential_providers(
                    maxResults=min(max_results, 100)
                )
                items = response.get("credentialProviders", [])
            elif provider_type == "workload":
                response = client.list_workload_identities(
                    maxResults=min(max_results, 100)
                )
                items = response.get("workloadIdentities", [])
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Unknown provider_type: {provider_type}. Valid: oauth2, api_key, workload"
                        }
                    ],
                }

            # Handle pagination
            next_token = response.get("nextToken")
            while next_token and len(items) < max_results:
                remaining = max_results - len(items)

                if verbose:
                    print(
                        f"📄 Fetching next page (total: {len(items)} so far)...",
                        flush=True,
                    )

                if provider_type == "oauth2":
                    response = client.list_oauth2_credential_providers(
                        maxResults=min(remaining, 20), nextToken=next_token
                    )
                    items.extend(response.get("credentialProviders", []))
                elif provider_type == "api_key":
                    response = client.list_api_key_credential_providers(
                        maxResults=min(remaining, 20), nextToken=next_token
                    )
                    items.extend(response.get("credentialProviders", []))
                elif provider_type == "workload":
                    response = client.list_workload_identities(
                        maxResults=min(remaining, 20), nextToken=next_token
                    )
                    items.extend(response.get("workloadIdentities", []))

                next_token = response.get("nextToken")

            if verbose:
                print(f"✅ Found {len(items)} {provider_type} providers", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"**Found {len(items)} {provider_type} providers:**"},
                    {"text": json.dumps(items, indent=2, default=str)},
                ],
            }

        elif action == "delete":
            if not name:
                return {
                    "status": "error",
                    "content": [{"text": "name is required for delete action"}],
                }

            if verbose:
                print(f"🗑️ Deleting {provider_type} provider: {name}", flush=True)

            if provider_type == "oauth2":
                client.delete_oauth2_credential_provider(name=name)
            elif provider_type == "api_key":
                client.delete_api_key_credential_provider(name=name)
            elif provider_type == "workload":
                client.delete_workload_identity(name=name)
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Unknown provider_type: {provider_type}. Valid: oauth2, api_key, workload"
                        }
                    ],
                }

            if verbose:
                print(f"✅ {provider_type} provider deleted", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": f"✅ **{provider_type.upper()} Provider Deleted**"},
                    {"text": f"**Name:** {name}"},
                ],
            }

        elif action == "update":
            if not name:
                return {
                    "status": "error",
                    "content": [{"text": "name is required for update action"}],
                }

            if verbose:
                print(f"🔄 Updating {provider_type} provider: {name}", flush=True)

            if provider_type == "oauth2":
                return _update_oauth2_provider(
                    client,
                    name,
                    client_id,
                    client_secret,
                    discovery_url,
                    authorization_endpoint,
                    token_endpoint,
                    scopes,
                    verbose,
                )
            elif provider_type == "api_key":
                return _update_api_key_provider(
                    client, name, api_key, header_name, verbose
                )
            elif provider_type == "workload":
                return _update_workload_identity(client, name, workload_arn, verbose)
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Unknown provider_type: {provider_type}. Valid: oauth2, api_key, workload"
                        }
                    ],
                }

        elif action == "get_vault":
            if verbose:
                print(f"🔐 Getting token vault configuration...", flush=True)

            response = client.get_token_vault()

            if verbose:
                print(f"✅ Retrieved token vault", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": "**Token Vault Configuration:**"},
                    {"text": json.dumps(response, indent=2, default=str)},
                ],
            }

        elif action == "set_vault_key":
            if not kms_key_id:
                return {
                    "status": "error",
                    "content": [{"text": "kms_key_id is required for set_vault_key"}],
                }

            if verbose:
                print(f"🔐 Setting token vault KMS key...", flush=True)

            client.set_token_vault_cmk(keyId=kms_key_id)

            if verbose:
                print(f"✅ Token vault KMS key set", flush=True)

            return {
                "status": "success",
                "content": [
                    {"text": "✅ **Token Vault KMS Key Set**"},
                    {"text": f"**Key ID:** {kms_key_id}"},
                ],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: create, get, list, delete, update, get_vault, set_vault_key"
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
                {"text": f"**Provider Type:** {provider_type}"},
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


# Helper functions for create operations
def _create_oauth2_provider(
    client: Any,
    name: str,
    vendor: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
    discovery_url: Optional[str],
    authorization_endpoint: Optional[str],
    token_endpoint: Optional[str],
    scopes: Optional[List[str]],
    verbose: bool,
) -> Dict[str, Any]:
    """Create OAuth2 credential provider."""
    if not all([vendor, client_id, client_secret]):
        return {
            "status": "error",
            "content": [
                {"text": "vendor, client_id, and client_secret required for OAuth2"}
            ],
        }

    if verbose:
        print(f"📝 Creating OAuth2 provider: {name} ({vendor})", flush=True)

    # Build provider config based on vendor
    oauth2_config = {}

    if vendor == "SlackOauth2":
        oauth2_config["slackOauth2ProviderConfig"] = {
            "clientId": client_id,
            "clientSecret": client_secret,
        }
    elif vendor == "GitHubOauth2":
        oauth2_config["githubOauth2ProviderConfig"] = {
            "clientId": client_id,
            "clientSecret": client_secret,
        }
    elif vendor == "GoogleOauth2":
        oauth2_config["googleOauth2ProviderConfig"] = {
            "clientId": client_id,
            "clientSecret": client_secret,
        }
    elif vendor == "CustomOauth2":
        if not discovery_url:
            return {
                "status": "error",
                "content": [{"text": "discovery_url required for CustomOauth2"}],
            }
        # Complex nested dict - mypy can't infer structure properly
        oauth2_config["customOauth2ProviderConfig"] = {
            "oauthDiscovery": {"discoveryUrl": discovery_url},  # type: ignore
            "clientId": client_id,
            "clientSecret": client_secret,
        }
    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown vendor: {vendor}. Valid: SlackOauth2, GitHubOauth2, GoogleOauth2, CustomOauth2"
                }
            ],
        }

    params = {
        "name": name,
        "credentialProviderVendor": vendor,
        "oauth2ProviderConfigInput": oauth2_config,
    }

    if verbose:
        print(f"🚀 Calling CreateOAuth2CredentialProvider API...", flush=True)

    response = client.create_oauth2_credential_provider(**params)

    if verbose:
        print(f"✅ OAuth2 provider created", flush=True)

    return {
        "status": "success",
        "content": [
            {"text": "✅ **OAuth2 Credential Provider Created**"},
            {"text": f"**Name:** {name}"},
            {"text": f"**Vendor:** {vendor}"},
            {
                "text": f"**Callback URL:** {response.get('callbackUrl', 'Not available')}"
            },
            {"text": f"**ARN:** {response.get('credentialProviderArn')}"},
        ],
    }


def _create_api_key_provider(
    client: Any,
    name: str,
    api_key: Optional[str],
    header_name: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """Create API key credential provider."""
    if not api_key:
        return {
            "status": "error",
            "content": [{"text": "api_key required for API key provider"}],
        }

    if verbose:
        print(f"📝 Creating API key provider: {name}", flush=True)

    params = {"name": name, "apiKey": api_key}

    if verbose:
        print(f"🚀 Calling CreateApiKeyCredentialProvider API...", flush=True)

    response = client.create_api_key_credential_provider(**params)

    if verbose:
        print(f"✅ API key provider created", flush=True)

    return {
        "status": "success",
        "content": [
            {"text": "✅ **API Key Credential Provider Created**"},
            {"text": f"**Name:** {name}"},
            {"text": f"**ARN:** {response.get('credentialProviderArn')}"},
        ],
    }


def _create_workload_identity(
    client: Any, name: str, workload_arn: Optional[str], verbose: bool
) -> Dict[str, Any]:
    """Create workload identity."""
    if verbose:
        print(f"📝 Creating workload identity: {name}", flush=True)

    params = {"name": name}
    if workload_arn:
        params["workloadArn"] = workload_arn

    if verbose:
        print(f"🚀 Calling CreateWorkloadIdentity API...", flush=True)

    response = client.create_workload_identity(**params)

    if verbose:
        print(f"✅ Workload identity created", flush=True)

    return {
        "status": "success",
        "content": [
            {"text": "✅ **Workload Identity Created**"},
            {"text": f"**Name:** {name}"},
            {"text": f"**ARN:** {response.get('workloadIdentityArn')}"},
        ],
    }


# Helper functions for update operations
def _update_oauth2_provider(
    client: Any,
    name: str,
    client_id: Optional[str],
    client_secret: Optional[str],
    discovery_url: Optional[str],
    authorization_endpoint: Optional[str],
    token_endpoint: Optional[str],
    scopes: Optional[List[str]],
    verbose: bool,
) -> Dict[str, Any]:
    """Update OAuth2 credential provider."""
    if verbose:
        print(f"📝 Updating OAuth2 provider: {name}", flush=True)

    params = {"name": name}

    if client_id:
        params["clientId"] = client_id
    if client_secret:
        params["clientSecret"] = client_secret

    # Build provider config if endpoints provided
    if discovery_url:
        params["oauthDiscovery"] = {"discoveryUrl": discovery_url}  # type: ignore[assignment]
    elif authorization_endpoint and token_endpoint:
        config = {
            "authorizationEndpoint": authorization_endpoint,
            "tokenEndpoint": token_endpoint,
        }
        if scopes:
            config["scopes"] = scopes  # type: ignore[assignment]
        params["genericOauth2ProviderConfig"] = config  # type: ignore[assignment]

    if verbose:
        print(f"🚀 Calling UpdateOAuth2CredentialProvider API...", flush=True)

    response = client.update_oauth2_credential_provider(**params)

    if verbose:
        print(f"✅ OAuth2 provider updated", flush=True)

    return {
        "status": "success",
        "content": [
            {"text": "✅ **OAuth2 Credential Provider Updated**"},
            {"text": f"**Name:** {name}"},
            {"text": json.dumps(response, indent=2, default=str)},
        ],
    }


def _update_api_key_provider(
    client: Any,
    name: str,
    api_key: Optional[str],
    header_name: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """Update API key credential provider."""
    if verbose:
        print(f"📝 Updating API key provider: {name}", flush=True)

    params = {"name": name}
    if api_key:
        params["apiKey"] = api_key
    if header_name:
        params["headerName"] = header_name

    if verbose:
        print(f"🚀 Calling UpdateApiKeyCredentialProvider API...", flush=True)

    response = client.update_api_key_credential_provider(**params)

    if verbose:
        print(f"✅ API key provider updated", flush=True)

    return {
        "status": "success",
        "content": [
            {"text": "✅ **API Key Credential Provider Updated**"},
            {"text": f"**Name:** {name}"},
            {"text": json.dumps(response, indent=2, default=str)},
        ],
    }


def _update_workload_identity(
    client: Any, name: str, workload_arn: Optional[str], verbose: bool
) -> Dict[str, Any]:
    """Update workload identity."""
    if verbose:
        print(f"📝 Updating workload identity: {name}", flush=True)

    params = {"name": name}
    if workload_arn:
        params["workloadArn"] = workload_arn

    if verbose:
        print(f"🚀 Calling UpdateWorkloadIdentity API...", flush=True)

    response = client.update_workload_identity(**params)

    if verbose:
        print(f"✅ Workload identity updated", flush=True)

    return {
        "status": "success",
        "content": [
            {"text": "✅ **Workload Identity Updated**"},
            {"text": f"**Name:** {name}"},
            {"text": json.dumps(response, indent=2, default=str)},
        ],
    }
