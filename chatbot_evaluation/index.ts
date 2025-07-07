import { MessagesAnnotation } from "@langchain/langgraph";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { myChatBot } from "./chatbot";
import { createSimulatedUser } from "./simulated";
import { StateGraph, END, START } from "@langchain/langgraph";

async function chatBotNode (state: typeof MessagesAnnotation.State) {
  const messages = state.messages
  const chatBotResponse = await myChatBot(messages);
  return { messages: [chatBotResponse] }
}

function swapRoles(messages: BaseMessage[]) {
  return messages.map((m) =>
    m instanceof AIMessage
      ? new HumanMessage({ content: m.content })
      : new AIMessage({ content: m.content }),
  )
}

async function simulatedUserNode (state: typeof MessagesAnnotation.State) {
  const messages = state.messages
  const newMessages = swapRoles(messages)
  // This returns a runnable directly, so we need to use `.invoke` below:
  const simulateUser = await createSimulatedUser();
  const response = await simulateUser.invoke({ messages: newMessages })

  return { messages: [{ role: "user", content: response.content }] }
}  

function shouldContinue(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  if (messages.length > 6) {
    return '__end__';
  } else if (messages[messages.length - 1].content === 'FINISHED') {
    return '__end__';
  } else {
    return 'continue';
  }
}

function createSimulation() {
  const workflow = new StateGraph(MessagesAnnotation)
    .addNode('user', simulatedUserNode)
    .addNode('chatbot', chatBotNode)
    .addEdge('chatbot', 'user')
    .addConditionalEdges('user', shouldContinue, {
      [END]: END,
      continue: 'chatbot',
    })
    .addEdge(START, 'chatbot')

  const simulation = workflow.compile()
  return simulation;
}

async function runSimulation() {
  const simulation = createSimulation()
  for await (const chunk of await simulation.stream({})) {
    const nodeName = Object.keys(chunk)[0];
    const messages = chunk[nodeName].messages;
    console.log(`${nodeName}: ${messages[0].content}`);
    console.log('\n---\n');
  }
}


await runSimulation();