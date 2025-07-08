import { RunnableConfig } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { GraphState } from "./state";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
dotenv.config();

const solvePrompt = ChatPromptTemplate.fromTemplate(
  `Solve the following task or problem. To solve the problem, we have made step-by-step Plan and
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:`,
);

export async function solve(state: typeof GraphState.State, config?: RunnableConfig) {
  const finalPlan = state.steps.reduce(
    (acc, step) => {
      const finalToolInput = step.toolInput.replace(
        /#E\d+/g,
        (match) => state.results[match] || match,
      );
      return acc + `Plan: ${step.plan}\n${step.variable} = ${step.tool}[${finalToolInput}]\n`;
    },
    ''
  );

  const model = new ChatOpenAI({
    temperature: 0,
    model: "gpt-4o",
  });
  const result = await solvePrompt
    .pipe(model)
    .invoke({ plan: finalPlan, task: state.task }, config);
  return {
    result: result.content.toString(),
  };
}