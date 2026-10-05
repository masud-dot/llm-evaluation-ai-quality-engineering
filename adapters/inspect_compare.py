"""Comparison only. Runs in the adapters environment."""
from inspect_ai import Task, eval as inspect_eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, tool


@tool
def reroute_shipment() -> Tool:
    async def execute(shipment_id: str, address_id: str) -> str:
        """Send a shipment to a different address.

        Args:
            shipment_id: Tidewater shipment identifier.
            address_id: Destination address identifier.
        """
        return "ok"
    return execute


@task
def reroute_task() -> Task:
    return Task(
        dataset=[Sample(input="Send TW-1042 to ADDR-2.",
                        target="rerouted")],
        solver=[use_tools(reroute_shipment()), generate()],
        scorer=includes())


scripted = [
    ModelOutput.for_tool_call(
        model="mockllm/model", tool_name="reroute_shipment",
        tool_arguments={"shipment_id": "TW-1042",
                        "address_id": "ADDR-2"}),
    ModelOutput.from_content(model="mockllm/model",
                             content="TW-1042 rerouted."),
]
log = inspect_eval(
    reroute_task(),
    model=get_model("mockllm/model", custom_outputs=scripted),
    display="none")[0]
samples = log.samples or []
calls = [(c.function, c.arguments)
         for m in samples[0].messages
         for c in (getattr(m, "tool_calls", None) or [])]
print(log.status)
for name, args in calls:
    print(name, args)
