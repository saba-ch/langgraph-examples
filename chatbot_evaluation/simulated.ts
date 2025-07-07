import { type Runnable } from "@langchain/core/runnables";
import { AIMessage, BaseMessageLike } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";

dotenv.config();

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

export async function createSimulatedUser(): Promise<Runnable<{ messages: BaseMessageLike[] }, AIMessage>> {
    const systemPromptTemplate = `You are a customer of an airline company. You are interacting with a user who is a customer support person 

{instructions}

If you have nothing more to add to the conversation, you must respond only with a single word: "FINISHED"`;

    const prompt = ChatPromptTemplate.fromMessages([
      ['system', systemPromptTemplate],
      ["placeholder", '{messages}'],
    ]);

    const instructions = `Your name is Harrison. You are trying to get a refund for the trip you took to Alaska.
You want them to give you ALL the money back. Be extremely persistent. This trip happened 5 years ago.`;

    const partialPrompt = await prompt.partial({ instructions });

    const simulatedUser = partialPrompt.pipe(llm);
    return simulatedUser;
}

