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
    """Manage OAuth2, API keys, and workload identities for AgentCore agents.

    Enables agents to securely access external services via OAuth2 flows or API keys.
    Credentials stored in AWS Secrets Manager with KMS encryption. Use @requires_access_token
    decorator in agent code for automatic token management.

    Quick Examples:
        # OAuth2 provider
        identity(
            action="create",
            provider_type="oauth2",
            name="slack-oauth",
            vendor="SlackOauth2",
            client_id="your-client-id",
            client_secret="your-secret"
        )

        # API key provider
        identity(
            action="create",
            provider_type="api_key",
            name="github-api",
            api_key="ghp_xxxxx"
        )

        # List providers
        identity(action="list", provider_type="oauth2")

    Args:
        action: Operation - create, get, list, delete, update, get_vault, set_vault_key
        name: Provider name (required for most operations)
        provider_type: oauth2 (default), api_key, or workload

        # OAuth2 specific
        vendor: SlackOauth2, GitHubOauth2, GoogleOauth2, or CustomOauth2
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret (stored encrypted)
        discovery_url: OAuth2 discovery URL (for CustomOauth2)
        authorization_endpoint: Auth endpoint (for CustomOauth2)
        token_endpoint: Token endpoint (for CustomOauth2)
        scopes: List of OAuth2 scopes

        # API Key specific
        api_key: API key value (stored encrypted)
        header_name: HTTP header name

        # Workload Identity specific
        workload_arn: Agent runtime ARN

        # Token Vault
        kms_key_id: KMS key for encryption (optional)

        # Common
        max_results: Max results for list (1-100, default: 20)
        region: AWS region (default: us-west-2)
        verbose: Enable verbose logging (default: False)

    Returns:
        Dict with status and content:
        {
            "status": "success|error",
            "content": [{"text": "Result message"}]
        }

    OAuth2 Vendors:
        - SlackOauth2: Slack OAuth with auto-discovery
        - GitHubOauth2: GitHub OAuth with auto-discovery
        - GoogleOauth2: Google OAuth with auto-discovery
        - CustomOauth2: Custom provider (needs discovery_url)

    IAM Permissions (Agent Execution Role):
        - bedrock-agentcore:GetResourceOauth2Token
        - secretsmanager:GetSecretValue

    IAM Permissions (Management Operations):
        - bedrock-agentcore-control:Create/Get/List/DeleteOAuth2CredentialProvider
        - bedrock-agentcore-control:Create/Get/List/DeleteApiKeyCredentialProvider
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
