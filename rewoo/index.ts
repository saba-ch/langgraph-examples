import { END, START, StateGraph } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph"; 
import { GraphState } from "./state";
import { getPlan } from "./planner";
import { toolExecution, _getCurrentTask } from "./executor";
import { solve } from "./solver";

const _route = (state: typeof GraphState.State) => {
  console.log("---ROUTE TASK---");
  const _step = _getCurrentTask(state);
  if (_step === null) {
    // We have executed all tasks
    return "solve";
  }
  // We are still executing tasks, loop back to the "tool" node
  return "tool";
};

const workflow = new StateGraph(GraphState)
  .addNode("plan", getPlan)
  .addNode("tool", toolExecution)
  .addNode("solve", solve)
  .addEdge("plan", "tool")
  .addEdge("solve", END)
  .addConditionalEdges("tool", _route)
  .addEdge(START, "plan");

// Compile
const app = workflow.compile({ checkpointer: new MemorySaver() });

const threadConfig = { configurable: { thread_id: "123" } };
let finalResult;
const stream = await app.stream(
  {
    task: "Where the winner of americas got talent 2023 is from?",
  },
  threadConfig,
);
for await (const item of stream) {
  console.log(item);
  console.log("-----");
  finalResult = item;
}