import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { RunnableConfig } from "@langchain/core/runnables";
import { GraphState } from "./state";
import dotenv from "dotenv";
dotenv.config();

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

const output = z.object({
  steps: z.array(z.object({
    plan: z.string(),
    variable: z.string(),
    tool: z.string(),
    toolInput: z.string(),
  })),
});

const structuredModel = model.withStructuredOutput(output);

const template =
  `For the following task, make plans that can solve the problem step by step. For each plan, indicate
which external tool together with tool input to retrieve evidence. You can store the evidence into a 
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pre-trained LLM like yourself. Useful when you need to act with general 
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.

For example,
Task: Hometown of the winner of the 2023 australian open
Plan: Search for the winner of the 2023 australian open
#E1 = Google[winner of the 2023 australian open]
Plan: Get the name of the winner
#E2 = LLM[What is the name of the winner, given #E1]
Plan: Get the hometown of the winner
#E3 = Google[hometown of the winner of the 2023 australian open]

Important!
Variables/results MUST be referenced using the # symbol!
The plan will be executed as a program, so no coreference resolution apart from naive variable replacement is allowed.
The ONLY way for steps to share context is by including #E<step> within the arguments of the tool.

Begin! 
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}`;

const promptTemplate = ChatPromptTemplate.fromMessages([["human", template]]);

export const planner = promptTemplate.pipe(structuredModel);

const task = "what is the hometown of the winner of the 2023 australian open?";
const result = await planner.invoke({ task });
console.log(result);

export async function getPlan(state: typeof GraphState.State, config?: RunnableConfig) {
  const task = state.task;
  const result = await planner.invoke({ task }, config);
  
  return result
}