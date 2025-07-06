import { ChatOpenAI } from "@langchain/openai";
import { Annotation, MessagesAnnotation, NodeInterrupt, StateGraph } from "@langchain/langgraph";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";
import { MemorySaver } from "@langchain/langgraph";
import dotenv from "dotenv";

dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
});

const checkpointer = new MemorySaver();

const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  nextRepresentative: Annotation<string>,
  refundAuthorized: Annotation<boolean>,
})

const initialSupport = async (state: typeof StateAnnotation.State) => {
  const SYSTEM_TEMPLATE =
    `You are frontline support staff for LangCorp, a company that sells computers.
Be concise in your responses.
You can chat with customers and help them with basic questions, but if the customer is having a billing or technical problem,
do not try to answer the question directly or gather information.
Instead, immediately transfer them to the billing or technical team by asking the user to hold for a moment.
Otherwise, just respond conversationally.`;

  const supportResponse = await model.invoke([
    new SystemMessage(SYSTEM_TEMPLATE),
    ...state.messages,
  ]);

  const CATEGORIZATION_SYSTEM_TEMPLATE = `You are an expert customer support routing system.
Your job is to detect whether a customer support representative is routing a user to a billing team or a technical team, or if they are just responding conversationally.`;
  const CATEGORIZATION_HUMAN_TEMPLATE =
    `The previous conversation is an interaction between a customer support representative and a user.
Extract whether the representative is routing the user to a billing or technical team, or whether they are just responding conversationally.
Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values:

If they want to route the user to the billing team, respond only with the word "BILLING".
If they want to route the user to the technical team, respond only with the word "TECHNICAL".
Otherwise, respond only with the word "RESPOND".`;

  const categorizationResponse = await model.invoke([
    new SystemMessage(CATEGORIZATION_SYSTEM_TEMPLATE),
    ...state.messages,
    new HumanMessage(CATEGORIZATION_HUMAN_TEMPLATE),
  ], {
    response_format: {
      type: 'json_schema',
      json_schema: {
        name: 'categorization',
        schema: z.object({
          nextRepresentative: z.enum(["BILLING", "TECHNICAL", "RESPOND"]),
        }),
      },
    },
  });

  const categorizationOutput = JSON.parse(categorizationResponse.content as string);

  return { messages: [supportResponse], nextRepresentative: categorizationOutput.nextRepresentative };
};

const billingSupport = async (state: typeof StateAnnotation.State) => {
  const SYSTEM_TEMPLATE =
    `You are an expert billing support specialist for LangCorp, a company that sells computers.
Help the user to the best of your ability, but be concise in your responses.
You have the ability to authorize refunds, which you can do by transferring the user to another agent who will collect the required information.
If you do, assume the other agent has all necessary information about the customer and their order.
You do not need to ask the user for more information.

Help the user to the best of your ability, but be concise in your responses.`;

  let trimmedHistory = state.messages;
  // Make the user's question the most recent message in the history.
  // This helps small models stay focused.
  if (trimmedHistory.at(-1) instanceof AIMessage) {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  const billingRepResponse = await model.invoke([
    new SystemMessage(SYSTEM_TEMPLATE),
    ...trimmedHistory,
  ]);

  const CATEGORIZATION_SYSTEM_TEMPLATE =
    `Your job is to detect whether a billing support representative wants to refund the user.`;
  const CATEGORIZATION_HUMAN_TEMPLATE =
    `The following text is a response from a customer support representative.
Extract whether they want to refund the user or not.
Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values:

If they want to refund the user, respond only with the word "REFUND".
Otherwise, respond only with the word "RESPOND".

Here is the text:

<text>
${billingRepResponse.content}
</text>.`;

  const categorizationResponse = await model.invoke([
    new SystemMessage(CATEGORIZATION_SYSTEM_TEMPLATE),
    ...trimmedHistory,
    new HumanMessage(CATEGORIZATION_HUMAN_TEMPLATE),
  ], {
    response_format: {
      type: 'json_schema',
      json_schema: {
        name: 'categorization',
        schema: z.object({
          nextRepresentative: z.enum(["REFUND", "RESPOND"]),
        }),
      },
    },
  });

  const categorizationOutput = JSON.parse(categorizationResponse.content as string);

  return { messages: billingRepResponse, nextRepresentative: categorizationOutput.nextRepresentative };
}

const technicalSupport = async (state: typeof StateAnnotation.State) => {
  const SYSTEM_TEMPLATE =
    `You are an expert at diagnosing technical computer issues. You work for a company called LangCorp that sells computers.
Help the user to the best of your ability, but be concise in your responses.`;

  const trimmedHistory = state.messages.at(-1) instanceof AIMessage ? state.messages.slice(0, -1) : state.messages;

  const response = await model.invoke([
    new SystemMessage(SYSTEM_TEMPLATE),
    ...trimmedHistory,
  ]);

  return { messages: response };
}

const handleRefund = async (state: typeof StateAnnotation.State) => {
  if(!state.refundAuthorized) {
    throw new NodeInterrupt("Human authorization required");
  }

  return {
    messages: {
      role: "assistant",
      content: "Refund processed!",
    },
  }
}

const initialSupportRouter = async (state: typeof StateAnnotation.State) => {
  switch(state.nextRepresentative) {
    case "BILLING":
      return "billing";
    case "TECHNICAL":
      return "technical";
    default:
      return "conversational";
  }
}

const billingSupportRouter = async (state: typeof StateAnnotation.State) => {
  switch(state.nextRepresentative) {
    case "REFUND":
      return "refund";
    default:
      return "__end__";
  }
}

const builder = new StateGraph(StateAnnotation)
  .addNode('initialSupport', initialSupport)
  .addNode('billingSupport', billingSupport)
  .addNode('technicalSupport', technicalSupport)
  .addNode('handleRefund', handleRefund)
  .addEdge('__start__', 'initialSupport')
  .addConditionalEdges(
    'initialSupport',
    initialSupportRouter,
    {
      billing: 'billingSupport',
      technical: 'technicalSupport',
      conversational: '__end__',
    }
  )
  .addConditionalEdges(
    'billingSupport',
    billingSupportRouter,
    {
      refund: 'handleRefund',
      __end__: '__end__',
    }
  )
  .addEdge('technicalSupport', '__end__')

const graph = builder.compile({
  checkpointer,
});

const stream = await graph.stream({
  messages: [
    {
      role: "user",
      content: "I've changed my mind and I want a refund for order #182818!",
    }
  ]
}, {
  configurable: {
    thread_id: "refund_testing_id",
  }
});

for await (const value of stream) {
  console.log("---STEP---");
  console.log(value);
  console.log("---END STEP---");
}


const currentState = await graph.getState({ configurable: { thread_id: "refund_testing_id" } });

console.log("CURRENT TASKS", JSON.stringify(currentState.tasks, null, 2));

console.log("NEXT TASKS", currentState.next);

await graph.updateState({ configurable: { thread_id: "refund_testing_id" } }, {
  refundAuthorized: true,
});

const resumedStream = await graph.stream(null, { configurable: { thread_id: "refund_testing_id" }});

for await (const value of resumedStream) {
  console.log(value);
}
