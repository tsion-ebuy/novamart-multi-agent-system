"""
NovaMart deployment pieces I wrote (excerpt): the Bedrock Guardrail, the
AgentCore Memory resource and the observability configuration.

Excerpt of src/agent_orchestrator.py. Not runnable on its own: it uses the
`config` module and AWS clients from the course starter.
"""
def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    topic_details = {
        "competitor products": (
            "Questions comparing NovaMart to competitor retailers, or asking about competitors' products or prices.",
            ["Is Amazon cheaper than NovaMart?", "Should I buy this from a competitor instead?"]),
        "pricing negotiations": (
            "Haggling, bargaining, or requests to change or lower an advertised price. Calculations using a price and discount the customer supplies are allowed and are not negotiation.",
            ["Can you knock 20% off if I buy two?", "Give me a better price or I walk."]),
        "legal threats": (
            "Threats of lawsuits, legal action, or involving lawyers or regulators against the company.",
            ["I will sue you if this is not refunded.", "My lawyer will be contacting you."]),
    }
    topics = []
    for topic_name in config.GUARDRAIL_BLOCKED_TOPICS:
        definition, examples = topic_details.get(
            topic_name, ("Requests about " + topic_name + ".", ["Tell me about " + topic_name + "."]))
        topics.append({"name": topic_name, "definition": definition,
                       "examples": examples, "type": "DENY"})

    params = dict(
        name=config.GUARDRAIL_NAME,
        description="NovaMart customer support safety guardrail",
        topicPolicyConfig={"topicsConfig": topics, "tierConfig": {"tierName": "STANDARD"}},
        contentPolicyConfig={"filtersConfig": [
            {"type": "SEXUAL",     "inputStrength": "HIGH",   "outputStrength": "HIGH"},
            {"type": "VIOLENCE",   "inputStrength": "HIGH",   "outputStrength": "HIGH"},
            {"type": "HATE",       "inputStrength": "HIGH",   "outputStrength": "HIGH"},
            {"type": "INSULTS",    "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
            {"type": "MISCONDUCT", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
        ]},
        wordPolicyConfig={"managedWordListsConfig": [{"type": "PROFANITY"}]},
        sensitiveInformationPolicyConfig={"piiEntitiesConfig": [
            {"type": "CREDIT_DEBIT_CARD_NUMBER",  "action": "BLOCK"},
            {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
            {"type": "EMAIL", "action": "ANONYMIZE"},
            {"type": "PHONE", "action": "ANONYMIZE"},
        ]},
        blockedInputMessaging="Sorry, I can't help with that request. I can help with orders, returns and NovaMart policies.",
        blockedOutputsMessaging="Sorry, I can't provide that response. Please ask about orders, returns or NovaMart policies.",
        crossRegionConfig={"guardrailProfileIdentifier": "us.guardrail.v1:0"},
    )
    try:
        response = bedrock_client.create_guardrail(**params)
    except ClientError as exc:
        # Untested fallback: some accounts require the content policy tier to match the topic tier.
        if "tier" not in str(exc).lower():
            raise
        print("  Retrying with the Standard tier on the content policy: " + exc.response["Error"]["Code"])
        params["contentPolicyConfig"]["tierConfig"] = {"tierName": "STANDARD"}
        response = bedrock_client.create_guardrail(**params)

    guardrail_id = response["guardrailId"]
    guardrail_version = None
    for _ in range(6):
        try:
            guardrail_version = bedrock_client.create_guardrail_version(
                guardrailIdentifier=guardrail_id,
                description="Published for AgentCore deployment",
            )["version"]
            break
        except Exception as exc:
            print("  Waiting for the guardrail to be ready (" + exc.__class__.__name__ + ")...")
            time.sleep(5)
    if guardrail_version is None:
        raise RuntimeError("Could not publish a numbered guardrail version")
    print(f"Guardrail created: {guardrail_id} (version: {guardrail_version})")
    return guardrail_id, guardrail_version


def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # Create AgentCore Memory
    # Use agentcore_control.create_memory() with:
    #   - name (memory_name) and a description
    #   - eventExpiryDuration = 7   (days)
    #   - memoryStrategies = [{'summaryMemoryStrategy': {
    #         'name': 'SessionSummary',
    #         'namespaces': ['/summaries/{actorId}/{sessionId}']}}]
    #   - clientToken (e.g. str(uuid.uuid4())) for idempotency
    # Store the API response in `response`.
    response = agentcore_control.create_memory(
        name=memory_name,
        description="NovaMart session-summary memory for multi-turn customer support conversations",
        eventExpiryDuration=7,
        memoryStrategies=[{'summaryMemoryStrategy': {
            'name': 'SessionSummary',
            'namespaces': ['/summaries/{actorId}/{sessionId}']}}],
        clientToken=str(uuid.uuid4()),
    )

    if response is None:
        raise NotImplementedError("configure_memory: create_memory() not implemented")

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    logging_configuration = {
        'cloudWatchConfig': {'logGroupName': config.AGENT_LOG_GROUP,
                             'logLevel': 'INFO', 'enabled': True},
        'xRayConfig':       {'enabled': True, 'samplingRate': 1.0},
    }
    try:
        summary = apply_observability_config(runtime_arn, logging_configuration)
        print(f"  CloudWatch logs: {config.AGENT_LOG_GROUP} (INFO)")
        print(f"  X-Ray sampling rate: {logging_configuration['xRayConfig']['samplingRate']:.0%}")
    except Exception as e:
        print(f"  [Note] Observability configuration failed: {e}")

