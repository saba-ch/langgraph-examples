import { HumanMessage } from "@langchain/core/messages";
import dotenv from "dotenv";
import { agentExecutor } from "./agent";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { JsonOutputToolsParser }from "@langchain/core/output_parsers/openai_tools"
import { END, START, StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { PlanExecuteState } from "./state";
import { planObject, planTool, responseTool } from "./tools";

dotenv.config();


const plannerPrompt = ChatPromptTemplate.fromTemplate(
  `For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

{objective}`,
);



const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const structuredModel = model.withStructuredOutput(planObject);

const planner = plannerPrompt.pipe(structuredModel);

const replannerPrompt = ChatPromptTemplate.fromTemplate(
  `For the given objective, come up with a simple step by step plan. 
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{pastSteps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that and use the 'response' function.
Otherwise, fill out the plan.  
Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.`,
);

const parser = new JsonOutputToolsParser();


const replanner = replannerPrompt
  .pipe(
    new ChatOpenAI({ model: "gpt-4o" }).bindTools([
      planTool,
      responseTool,
    ]),
  )
  .pipe(parser);

  async function executeStep(
    state: typeof PlanExecuteState.State,
    config?: RunnableConfig,
  ): Promise<Partial<typeof PlanExecuteState.State>> {
    console.log("Executing step");
    const task = state.plan[0];
    const input = {
      messages: [new HumanMessage(task)],
    };
    const { messages } = await agentExecutor.invoke(input, config);
  
    return {
      pastSteps: [[task, messages[messages.length - 1].content.toString()]],
      plan: state.plan.slice(1),
    };
  }
  
  async function planStep(
    state: typeof PlanExecuteState.State,
  ): Promise<Partial<typeof PlanExecuteState.State>> {
    console.log("Planning step");
    const plan = await planner.invoke({ objective: state.input });
    return { plan: plan.steps };
  }
  
  async function replanStep(
    state: typeof PlanExecuteState.State,
  ): Promise<Partial<typeof PlanExecuteState.State>> {
    console.log("Replanning step");
    const output = await replanner.invoke({
      input: state.input,
      plan: state.plan.join("\n"),
      pastSteps: state.pastSteps
        .map(([step, result]) => `${step}: ${result}`)
        .join("\n"),
    });
    const toolCall = output[0];
  
    if (toolCall.type == "response") {
      return { response: toolCall.args?.response };
    }
  
    return { plan: toolCall.args?.steps };
  }
  
  function shouldEnd(state: typeof PlanExecuteState.State) {
    return state.response ? "true" : "false";
  }
  
  const workflow = new StateGraph(PlanExecuteState)
    .addNode("planner", planStep)
    .addNode("agent", executeStep)
    .addNode("replan", replanStep)
    .addEdge(START, "planner")
    .addEdge("planner", "agent")
    .addEdge("agent", "replan")
    .addConditionalEdges("replan", shouldEnd, {
      true: END,
      false: "agent",
    });
  
  // Finally, we compile it!
  // This compiles it into a LangChain Runnable,
  // meaning you can use it as you would any other runnable
  const app = workflow.compile();
  

const config = { recursionLimit: 50 };
const inputs = {
  input: "what is the hometown of the 2024 Australian open winner?",
};

for await (const event of await app.stream(inputs, config)) {
  console.log(event);
}