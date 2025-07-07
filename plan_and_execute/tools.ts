import { TavilySearch } from "@langchain/tavily";
import dotenv from "dotenv";
import { z } from "zod";
import { tool } from "@langchain/core/tools";

dotenv.config();

export const tools = [new TavilySearch({ maxResults: 3 })];

export const planObject = z.object({
  steps: z
    .array(z.string()).describe("different steps to follow, should be in sorted order"),
})

export const responseObject = z.object({
  response: z.string().describe("Response to user."),
})

export const responseTool = tool(() => {}, {
  name: "response",
  description: "Respond to the user.",
  schema: responseObject,
})

export const planTool = tool(() => {}, {
  name: "plan",
  description: "This tool is used to plan the steps to follow.",
  schema: planObject,
})
