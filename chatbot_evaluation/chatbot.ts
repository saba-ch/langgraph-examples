import { ChatOpenAI } from '@langchain/openai'
import type { AIMessageChunk, BaseMessageLike } from "@langchain/core/messages";
import dotenv from "dotenv";

dotenv.config();

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

export async function myChatBot(messages: BaseMessageLike[]): Promise<AIMessageChunk> {
  const systemMessage = {
    role: 'system',
    content: 'You are a customer support agent for an airline.',
  };
  const allMessages = [systemMessage, ...messages];

  const response = await llm.invoke(allMessages)
  return response
}


