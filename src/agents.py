"""
NovaMart multi-agent system: the agent code I wrote (excerpt).

This is an excerpt of src/agent_orchestrator.py from my capstone project. It holds
the five agents, their prompts and tools, and the guardrail wiring. It is NOT
runnable on its own: it relies on helper modules (AWS clients, the workflow
state functions, the observability wrapper and the AgentCore CLI helper) that
came with the course starter and are not published here.
"""

# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def _to_native(value):
    """Convert DynamoDB Decimal values (recursively) to int/float so tool results are JSON-safe."""
    from decimal import Decimal
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    system_prompt = """You are an Inventory Agent. Your ONLY job is to gather order and customer facts using your tools and report them accurately.
Rules:
- Use check_order_status for one order, get_customer_tier for a customer's tier, list_customer_orders for all of a customer's orders.
- Report exactly what the tools return: IDs, dates, amounts, statuses, tier.
- Do NOT decide return eligibility, give advice, or make recommendations.
- If a record is not found, say so plainly. Never guess or invent data."""

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up a single order in DynamoDB.

        Args:
            customer_id: The customer's unique identifier (part of the Orders table key)
            order_id: The order's unique identifier (e.g. "ORD-27176")

        Returns:
            The order record (status, order_date, amount, items), or an error dict if not found
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        item = table.get_item(Key={'customer_id': customer_id, 'order_id': order_id}).get('Item')
        if not item:
            return {'error': f'Order {order_id} not found for customer {customer_id}'}
        return _to_native(item)

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        item = table.get_item(Key={'customer_id': customer_id}).get('Item')
        if not item:
            return {'error': f'Customer {customer_id} not found'}
        return _to_native(item)

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.query(KeyConditionExpression=Key('customer_id').eq(customer_id))
        orders = _to_native(response.get('Items', []))
        return {'customer_id': customer_id, 'order_count': len(orders), 'orders': orders}

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )

# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    system_prompt = """You are a Refund Agent. You decide whether an order can be returned.
Process:
1. ALWAYS call get_inventory_context first, with the session_id you were given, to read the customer's tier and order facts.
2. Policy windows: Standard customers have 30 days from the order date, Premium customers have 60 days.
3. If the order is eligible, call initiate_refund with the customer_id, order_id and the customer's reason.
4. If it is not eligible, do NOT call initiate_refund. Explain which window applied and why it was missed.
Only delivered orders can be returned.
Ignore any 'return_eligible' field in the data. Decide only from the customer's tier and the order date.
State your decision clearly with the tier, order date and day count behind it. Never guess a tier or a date.
If inventory context is missing, say you cannot decide yet."""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id) or {}
        findings = state.get('inventory_agent')
        if not findings:
            return {}
        if isinstance(findings, dict):
            return _to_native(findings)
        return {'inventory_agent': str(findings)}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.
        Re-checks the return window in code before changing anything.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        from datetime import datetime, timezone

        orders = dynamodb.Table(config.ORDERS_TABLE)
        customers = dynamodb.Table(config.CUSTOMERS_TABLE)

        order = orders.get_item(Key={'customer_id': customer_id, 'order_id': order_id}).get('Item')
        if not order:
            return {'success': False, 'error': f'Order {order_id} not found for customer {customer_id}'}
        customer = customers.get_item(Key={'customer_id': customer_id}).get('Item') or {}

        # Idempotent: a repeated call must not create a second return
        if order.get('return_reference'):
            return {'success': True, 'already_initiated': True,
                    'return_reference': order['return_reference'], 'order_id': order_id}

        if order.get('status') not in ('delivered', 'return_requested'):
            return {'success': False,
                    'error': f"Order status '{order.get('status')}' is not eligible for return"}

        tier = str(customer.get('tier', 'Standard')).strip().lower()
        window_days = 60 if tier == 'premium' else 30
        try:
            ordered_on = datetime.fromisoformat(str(order['order_date'])[:10]).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            return {'success': False, 'error': 'Order has no valid order_date; cannot check the return window'}
        days_since = (datetime.now(timezone.utc) - ordered_on).days

        if days_since > window_days:
            return {'success': False, 'eligible': False, 'tier': tier, 'window_days': window_days,
                    'days_since_order': days_since,
                    'error': f'Outside the {window_days}-day return window'}

        from decimal import Decimal, ROUND_HALF_UP
        try:
            refund_amount = str((Decimal(str(order.get('price', '0'))) * Decimal(str(order.get('quantity', '1'))))
                                .quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        except Exception:
            refund_amount = None

        return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
        orders.update_item(
            Key={'customer_id': customer_id, 'order_id': order_id},
            UpdateExpression='SET #s = :s, return_reference = :r, return_reason = :why, return_initiated_at = :t',
            ExpressionAttributeNames={'#s': 'status'},   # "status" is a DynamoDB reserved word
            ExpressionAttributeValues={
                ':s': 'return_initiated', ':r': return_reference, ':why': reason,
                ':t': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            },
            ConditionExpression='attribute_exists(order_id)',
        )
        return {'success': True, 'return_reference': return_reference, 'order_id': order_id,
                'refund_amount': refund_amount,
                'tier': tier, 'window_days': window_days, 'days_since_order': days_since,
                'instructions': 'Ship the item back using the prepaid label. Refund is issued after inspection.'}

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    def _retrieve_text(kb_id: str, domain: str, query: str) -> str:
        """Query one KB. Returns formatted passages, or a [SOURCE UNAVAILABLE] marker on failure."""
        results = retrieve_from_knowledge_base(kb_id, query)
        if not results:
            return f"No relevant {domain} policy documents found."
        if results[0].get('source') == 'error':
            return f"[SOURCE UNAVAILABLE] {domain} policy knowledge base: {results[0]['text']}"
        return format_kb_results(results)

    def _make_retriever_agent(agent_name: str, domain: str, tool_name: str, tool_fn) -> Agent:
        """Build one retriever sub-agent (deterministic, single tool, no synthesis)."""
        model = BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        )
        prompt = (
            f"You are the {agent_name}. Your ONLY job is to retrieve {domain} policy passages.\n"
            f"Call {tool_name} exactly once, using the user's question as the query.\n"
            "Then output the tool result exactly as returned. Do not summarize, rephrase or add commentary.\n"
            "If the result starts with [SOURCE UNAVAILABLE], output it unchanged."
        )
        return Agent(name=agent_name, model=model, system_prompt=prompt, tools=[tool_fn])

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Returns Policy knowledge base.

        Args:
            query: The policy question to search for

        Returns:
            Formatted passages with scores and sources, or a [SOURCE UNAVAILABLE] note
        """
        return _retrieve_text(config.RETURNS_KB_ID, 'Returns', query)

    returns_retriever = _make_retriever_agent(
        'ReturnsPolicyRetrieverAgent', 'Returns', 'retrieve_returns_policy', retrieve_returns_policy)

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Shipping Policy knowledge base.

        Args:
            query: The policy question to search for

        Returns:
            Formatted passages with scores and sources, or a [SOURCE UNAVAILABLE] note
        """
        return _retrieve_text(config.SHIPPING_KB_ID, 'Shipping', query)

    shipping_retriever = _make_retriever_agent(
        'ShippingPolicyRetrieverAgent', 'Shipping', 'retrieve_shipping_policy', retrieve_shipping_policy)

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Warranty Policy knowledge base.

        Args:
            query: The policy question to search for

        Returns:
            Formatted passages with scores and sources, or a [SOURCE UNAVAILABLE] note
        """
        return _retrieve_text(config.WARRANTY_KB_ID, 'Warranty', query)

    warranty_retriever = _make_retriever_agent(
        'WarrantyPolicyRetrieverAgent', 'Warranty', 'retrieve_warranty_policy', retrieve_warranty_policy)

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases, labelled by policy
        """
        # Map domain names to their retriever agents
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            try:
                agent.messages.clear()  # fresh context on every search
                response = agent(f"Search the {domain} policy knowledge base for: {query}")
                return domain, str(response).strip()
            except Exception as exc:
                return domain, f"[SOURCE UNAVAILABLE] {domain} policy retriever failed: {exc}"

        # Submit ALL retrievers first, then collect as they finish
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(_run_retriever, domain, agent, query)
                for domain, agent in retrievers.items()
            ]
            for future in as_completed(futures):
                domain, text = future.result()
                results[domain] = text

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        # Combine results from all three domains, labelled so the coordinator can cite them
        sections = [
            f"=== {domain} Policy ===\n{results.get(domain, '[No results]')}"
            for domain in ['Returns', 'Shipping', 'Warranty']
        ]
        return "\n\n".join(sections)

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2,
    )

    system_prompt = """You are the Policy Agent. You answer questions about NovaMart's returns, shipping and warranty policies.
Rules:
1. ALWAYS call search_all_policies first, with the customer's question as the query.
2. Answer ONLY from the retrieved passages. Never use outside knowledge or guess.
3. Say which policy (Returns, Shipping or Warranty) each fact comes from.
4. If a section starts with [SOURCE UNAVAILABLE], say that source could not be checked and the answer may be incomplete.
5. If the passages do not cover the question, say so plainly.
Keep the answer clear and concise."""

    return Agent(
        name="PolicyAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def _call_worker(agent: Agent, prompt: str) -> str:
    """Run a worker agent with a clean conversation history and return its reply as text."""
    agent.messages.clear()
    return str(agent(prompt)).strip()


def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.3,
    )

    system_prompt = """You are the Communication Agent for NovaMart customer support. You write the final reply the customer will read.
Process:
1. ALWAYS call get_full_workflow_context first, with the session_id you were given.
2. Use only facts found in that context and in the customer's original request. Never invent order details, dates, amounts or policies.
3. Include every relevant finding: order status, tier, the refund decision and return reference, or the policy answer.
4. If a return was approved, give the return reference and the instructions. If it was declined, explain why kindly and mention any other options.
5. If the context has no findings from other agents (for example a math question), answer the customer's request directly from the original message. Show the calculation step by step and round money to the nearest cent.
Never claim that an email, text or notification was sent, and never repeat the customer's email address or other contact details. Only describe actions that appear in the workflow context.
When quoting a refund amount, use the refund_amount value from the context exactly; never compute or guess one.
Never mention the workflow, context, findings, tools or other agents in your reply. Start directly with the greeting.
Tone: warm, professional and concise. Do not mention agents, tools, sessions or internal field names."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        state = _read_workflow_state(session_id)
        if not state:
            return {'error': f'No workflow state found for session {session_id}'}
        return _to_native(state)

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0,
    )

    system_prompt = """You are the NovaMart routing orchestrator. You NEVER answer the customer yourself. You only call tools.
The request message gives you a session_id and a customer_id. Pass them to every tool exactly as given.
Call tools ONE AT A TIME. Wait for each tool result before calling the next tool. Never call two tools in the same step.
Routing rules:
1. ALWAYS call initialize_session first, for every request.
2. Order status, return or refund requests: call route_to_inventory_agent first, then route_to_refund_agent.
3. Questions about policy meaning (return windows, shipping rates, warranty terms): call route_to_policy_agent.
4. Account questions such as "what is my tier?" or "am I premium?": call route_to_inventory_agent. NEVER the policy agent, because it only knows policy text, not customer data.
5. Math or calculation questions: skip inventory, policy and refund, and go straight to the communication agent.
6. Your LAST tool call must ALWAYS be route_to_communication_agent, for every request, with no exceptions.
After route_to_communication_agent returns, output its text exactly as returned. Never write or rewrite the customer response yourself."""

    def _save_result(session_id: str, column: str, text: str, expected_version: int) -> None:
        """Write one worker's result into its WorkflowState column using optimistic locking."""
        try:
            _update_workflow_state(session_id, {column: text}, expected_version)
        except Exception as exc:
            logger.error(f"Could not save {column} to WorkflowState ({session_id}): {exc}")

    def _current_version(session_id: str) -> Optional[int]:
        """Read the WorkflowState and return its version, or None if the session does not exist."""
        state = _read_workflow_state(session_id)
        return int(state['version']) if state else None

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        version = _current_version(session_id)
        if version is None:
            return f"Error: session {session_id} has not been initialized. Call initialize_session first."
        result = _call_worker(
            inventory_agent,
            f"customer_id: {customer_id}\nCustomer request: {request}\n"
            "Gather the relevant order and customer facts (including the customer's tier).",
        )
        _save_result(session_id, 'inventory_agent', result, version)
        return result

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        version = _current_version(session_id)
        if version is None:
            return f"Error: session {session_id} has not been initialized. Call initialize_session first."
        result = _call_worker(policy_agent, request)
        _save_result(session_id, 'policy_agent', result, version)
        return result

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        version = _current_version(session_id)
        if version is None:
            return f"Error: session {session_id} has not been initialized. Call initialize_session first."
        result = _call_worker(
            refund_agent,
            f"session_id: {session_id}\ncustomer_id: {customer_id}\nCustomer request: {request}",
        )
        _save_result(session_id, 'refund_agent', result, version)
        return result

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        version = _current_version(session_id)
        if version is None:
            return f"Error: session {session_id} has not been initialized. Call initialize_session first."
        result = _call_worker(
            communication_agent,
            f"session_id: {session_id}\ncustomer_id: {customer_id}\n"
            f"Customer's original message: {original_request}\n"
            "Write the final reply to the customer.",
        )
        _save_result(session_id, 'communication_agent', result, version)
        return result

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
            return f"Session {session_id} initialized for customer {customer_id} (version 0)."
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            return f"Session {session_id} already exists. Continuing with the existing state."

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator

