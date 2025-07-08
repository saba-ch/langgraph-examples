import { TavilySearch } from "@langchain/tavily";
import { GraphState } from "./state";
import { RunnableConfig } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
dotenv.config();

const search = new TavilySearch();
const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

export const _getCurrentTask = (state: typeof GraphState.State) => {
  if (!state.results) return 1;
  if (Object.entries(state.results).length === state.steps.length) return null;
  return Object.entries(state.results).length + 1;
};

export async function toolExecution(state: typeof GraphState.State, config?: RunnableConfig) {
  console.log("---EXECUTE TOOL---");
  const _step = _getCurrentTask(state);
  if (!_step) throw new Error("No current task found");

  const {
    variable,
    tool,
    toolInput,
  } = state.steps[_step - 1];

  
  let formattedToolInput = toolInput;
  const _results = state.results || {};
  for (const [k, v] of Object.entries(_results)) {
    formattedToolInput = formattedToolInput.replace(k, v);
  }

  switch (tool) {
    case "Google":
      const googleResult = await search.invoke({ query: formattedToolInput }, config);
      _results[variable] = JSON.stringify(googleResult, null, 2);
      break;
    case "LLM":
      const llmResult = await model.invoke(formattedToolInput, config);
      _results[variable] = llmResult.content
      break;
    default:
      throw new Error("Invalid tool specified");
  }
  return { results: _results };
}