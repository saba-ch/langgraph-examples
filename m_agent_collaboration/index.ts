import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { Runnable, RunnableConfig } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { END, START, StateGraph } from "@langchain/langgraph";

dotenv.config();
import { createAgent } from "./agent";
import { chartTool, tavilyTool } from "./tools";
import { AgentState } from "./state";

// Helper function to run a node for a given agent
async function runAgentNode(props: {
  state: typeof AgentState.State;
  agent: Runnable;
  name: string;
  config?: RunnableConfig;
}) {
  const { state, agent, name, config } = props;
  let result = await agent.invoke(state, config);
  // We convert the agent output into a format that is suitable
  // to append to the global state
  if (!result?.tool_calls || result.tool_calls.length === 0) {
    // If the agent is NOT calling a tool, we want it to
    // look like a human message.
    result = new HumanMessage({ ...result, name: name });
  }
  return {
    messages: [result],
    // Since we have a strict workflow, we can
    // track the sender so we know who to pass to next.
    sender: name,
  };
}

const llm = new ChatOpenAI({ modelName: "gpt-4o" });

// Research agent and node
const researchAgent = await createAgent({
  llm,
  tools: [tavilyTool],
  systemMessage:
    "You should provide accurate data for the chart generator to use.",
});

async function researchNode(
  state: typeof AgentState.State,
  config?: RunnableConfig,
) {
  return runAgentNode({
    state: state,
    agent: researchAgent,
    name: "Researcher",
    config,
  });
}

// Chart Generator
const chartAgent = await createAgent({
  llm,
  tools: [chartTool],
  systemMessage: "Any charts you display will be visible by the user.",
});

async function chartNode(state: typeof AgentState.State) {
  return runAgentNode({
    state: state,
    agent: chartAgent,
    name: "ChartGenerator",
  });
}

const tools = [tavilyTool, chartTool];
// This runs tools in the graph
const toolNode = new ToolNode<typeof AgentState.State>(tools);

function router(state: typeof AgentState.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  if (lastMessage?.tool_calls && lastMessage.tool_calls.length > 0) {
    // The previous agent is invoking a tool
    return "call_tool";
  }
  if (
    typeof lastMessage.content === "string" &&
    lastMessage.content.includes("FINAL ANSWER")
  ) {
    // Any agent decided the work is done
    return "end";
  }
  return "continue";
}


// 1. Create the graph
const workflow = new StateGraph(AgentState)
   // 2. Add the nodes; these will do the work
  .addNode("Researcher", researchNode)
  .addNode("ChartGenerator", chartNode)
  .addNode("call_tool", toolNode);

// 3. Define the edges. We will define both regular and conditional ones
// After a worker completes, report to supervisor
workflow.addConditionalEdges("Researcher", router, {
  // We will transition to the other agent
  continue: "ChartGenerator",
  call_tool: "call_tool",
  end: END,
});

workflow.addConditionalEdges("ChartGenerator", router, {
  // We will transition to the other agent
  continue: "Researcher",
  call_tool: "call_tool",
  end: END,
});

workflow.addConditionalEdges(
  "call_tool",
  // Each agent node updates the 'sender' field
  // the tool calling node does not, meaning
  // this edge will route back to the original agent
  // who invoked the tool
  (x) => x.sender,
  {
    Researcher: "Researcher",
    ChartGenerator: "ChartGenerator",
  },
);

workflow.addEdge(START, "Researcher");
const graph = workflow.compile();

const streamResults = await graph.stream(
  {
    messages: [
      new HumanMessage({
        content: "Generate a bar chart of the US gdp over the past 3 years.",
      }),
    ],
  },
  { recursionLimit: 150 },
);

const prettifyOutput = (output: Record<string, any>) => {
  const keys = Object.keys(output);
  const firstItem = output[keys[0]];

  if ("messages" in firstItem && Array.isArray(firstItem.messages)) {
    const lastMessage = firstItem.messages[firstItem.messages.length - 1];
    console.dir({
      type: lastMessage._getType(),
      content: lastMessage.content,
      tool_calls: lastMessage.tool_calls,
    }, { depth: null });
  }

  if ("sender" in firstItem) {
    console.log({
      sender: firstItem.sender,
    })
  }
}

for await (const output of await streamResults) {
  if (!output?.__end__) {
    prettifyOutput(output);
    console.log("----");
  }
}