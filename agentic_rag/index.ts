import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { BaseMessage, AIMessage, HumanMessage, AIMessageChunk } from "@langchain/core/messages";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { createRetrieverTool } from "langchain/tools/retriever";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { pull } from "langchain/hub";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";

dotenv.config();

const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

const docs = await Promise.all(
  urls.map((url) => new CheerioWebBaseLoader(url).load()),
);
const docsList = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const embedder = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
});

const docSplits = await textSplitter.splitDocuments(docsList);

// Add to vectorDB
const vectorStore = await MemoryVectorStore.fromDocuments(
  docSplits,
  embedder,
);

const retriever = vectorStore.asRetriever();

const GraphState = Annotation.Root({
  messages: Annotation<(BaseMessage | AIMessage)[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  })
})

const tool = createRetrieverTool(
  retriever,
  {
    name: "retrieve_blog_posts",
    description:
      "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
  }
);

const tools = [tool];
const toolNode = new ToolNode<typeof GraphState.State>(tools);

const shouldRetrieve = (state: typeof GraphState.State) => {
  const lastMessage = state.messages.at(-1);

  if(lastMessage instanceof AIMessageChunk && lastMessage.tool_calls?.length) {
    return 'retrieve';
  }

  return END;
}

const gradeDocuments = async (state: typeof GraphState.State) => {
  const relevanceTool = {
    name: 'give_relevance_score',
    description: 'Give a relevance score to the documents',
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    })
  }

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of retrieved docs to a user question.
Here are the retrieved docs:
\n ------- \n
{context} 
\n ------- \n
Here is the user question: {question}
If the content of the docs are relevant to the users question, score them as relevant only if they can answer the user question.
Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
Yes: The docs are relevant to the question. 
No: The docs are not relevant to the question.`,
  );

  const gradeModel = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
  })
    .bindTools(
      [relevanceTool],
      {
        tool_choice: relevanceTool.name,
      }
    );

  const chain = prompt.pipe(gradeModel)

  const score = await chain.invoke({
    question: state.messages[0]?.content,
    context: state.messages.at(-1)?.content,
  });

  return {
    messages: [score]
  }
}

const checkRelevance = async (state: typeof GraphState.State) => {
  const lastMessage = state.messages.at(-1);

  if(!(lastMessage instanceof AIMessage)) throw new Error("The 'checkRelevance' node requires the most recent message to contain tool calls.");

  const toolCall = lastMessage.tool_calls?.[0];
  if(!toolCall) throw new Error("Last message was not a function message");

  return toolCall.args.binaryScore === 'yes' ? 'yes' : 'no';
}

const agent = async (state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> => {
  const { messages } = state;
  // Find the AIMessage which contains the `give_relevance_score` tool call,
  // and remove it if it exists. This is because the agent does not need to know
  // the relevance score.
  const filteredMessages = messages.filter((message) => {
    if(!(message instanceof AIMessageChunk || message instanceof AIMessage)) return true;
    const toolCall = message.tool_calls?.[0];
    if(!toolCall || toolCall.name !== "give_relevance_score") return true;

    return false;
  });

  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  }).bindTools(tools);

  const response = await model.invoke(filteredMessages);
  return {
    messages: [response],
  };
}

const rewrite = async (state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> => {
  const question = state.messages[0]?.content;

  const prompt = ChatPromptTemplate.fromTemplate(
    `Look at the input and try to reason about the underlying semantic intent / meaning. \n 
Here is the initial question:
\n ------- \n
{question} 
\n ------- \n
Formulate an improved question:`,
  );

  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
  });
  const response = await prompt.pipe(model).invoke({ question });
  return {
    messages: [response],
  };
}

const generate = async (state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> => {
  const question = state.messages[0]?.content;

  const lastToolMessage = state.messages.slice().reverse().find((msg) => msg.getType() === "tool");
  if (!lastToolMessage) {
    throw new Error("No tool message found in the conversation history");
  }

  const docs = lastToolMessage.content as string;

  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

  const llm = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  });

  const ragChain = prompt.pipe(llm);

  const response = await ragChain.invoke({
    context: docs,
    question,
  });

  return {
    messages: [response],
  };
}


const workflow = new StateGraph(GraphState)
  .addNode('agent', agent)
  .addNode('retrieve', toolNode)
  .addNode('gradeDocuments', gradeDocuments)
  .addNode('rewrite', rewrite)
  .addNode('generate', generate)
  .addEdge(START, 'agent')
  .addConditionalEdges(
    'agent',
    shouldRetrieve
  )
  .addEdge('retrieve', 'gradeDocuments')
  .addConditionalEdges(
    'gradeDocuments',
    checkRelevance,
    {
      'yes': 'generate',
      'no': 'rewrite',
    }
  )
  .addEdge('rewrite', 'agent')
  .addEdge('generate', END);

const app = await workflow.compile();

const inputs = {
  messages: [
    new HumanMessage(
      "What are the types of agent memory based on Lilian Weng's blog post?",
    ),
  ],
};
let finalState;
for await (const output of await app.stream(inputs)) {
  for (const [key, value] of Object.entries(output)) {
    const lastMsg = output[key].messages[output[key].messages.length - 1];
    console.log(`Output from node: '${key}'`);
    console.dir({
      type: lastMsg._getType(),
      content: lastMsg.content,
      tool_calls: lastMsg.tool_calls,
    }, { depth: null });
    console.log("---\n");
    finalState = value;
  }
}

console.log(JSON.stringify(finalState, null, 2));