import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

import { tools } from "./tools";

export const agentExecutor = createReactAgent({
  llm: new ChatOpenAI({ model: "gpt-4o" }),
  tools: tools,
});