import 'cheerio'
import { z } from "zod";
import { HumanMessage, BaseMessage, SystemMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { Runnable } from "@langchain/core/runnables";
import { StructuredToolInterface } from "@langchain/core/tools";
import { MessagesAnnotation } from "@langchain/langgraph";
import { JsonOutputToolsParser }from "@langchain/core/output_parsers/openai_tools"

export const agentStateModifier = (
  systemPrompt: string,
  tools: StructuredToolInterface[],
  teamMembers: string[],
): ((state: typeof MessagesAnnotation.State) => BaseMessage[]) => {
  const toolNames = tools.map((t) => t.name).join(", ");
  const systemMsgStart = new SystemMessage(systemPrompt +
    "\nWork autonomously according to your specialty, using the tools available to you." +
    " Do not ask for clarification." +
    " Your other team members (and other teams) will collaborate with you with their own specialties." +
    ` You are chosen for a reason! You are one of the following team members: ${teamMembers.join(", ")}.`)
  const systemMsgEnd = new SystemMessage(`Supervisor instructions: ${systemPrompt}\n` +
      `Remember, you individually can only use these tools: ${toolNames}` +
      "\n\nEnd if you have already completed the requested task. Communicate the work completed.");

  return (state: typeof MessagesAnnotation.State): any[] => 
    [systemMsgStart, ...state.messages, systemMsgEnd];
}

export async function runAgentNode(params: {
  state: any;
  agent: Runnable;
  name: string;
}) {
  const { state, agent, name } = params;
  const result = await agent.invoke({
    messages: state.messages,
  });
  const lastMessage = result.messages[result.messages.length - 1];
  return {
    messages: [new HumanMessage({ content: lastMessage.content, name })],
  };
}

export async function createTeamSupervisor(
  llm: ChatOpenAI,
  systemPrompt: string,
  members: string[],
): Promise<Runnable> {
  const options = ["FINISH", ...members];
  const routeTool = {
    name: "route",
    description: "Select the next role.",
    schema: z.object({
      reasoning: z.string(),
      next: z.enum(["FINISH", ...members]),
      instructions: z.string().describe("The specific instructions of the sub-task the next role should accomplish."),
    })
  }
  let prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("messages"),
    [
      "system",
      "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
    ],
  ]);
  prompt = await prompt.partial({
    options: options.join(", "),
    team_members: members.join(", "),
  });

  const supervisor = prompt
    .pipe(
      llm.bindTools([routeTool], {
        tool_choice: "route",
      }),
    )
    .pipe(new JsonOutputToolsParser())
    // select the first one
    .pipe((x) => ({
      next: x[0].args.next,
      instructions: x[0].args.instructions,
    }));

  return supervisor;
}

