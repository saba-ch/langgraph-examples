import 'cheerio'
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Annotation } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { agentStateModifier, createTeamSupervisor, runAgentNode } from "./agent";
import { tavilyTool, scrapeWebpage, createOutlineTool, readDocumentTool, writeDocumentTool, editDocumentTool, chartTool } from "./tools";
import { END, START, StateGraph } from "@langchain/langgraph";
import { RunnableLambda } from "@langchain/core/runnables";
import * as fs from "fs/promises";
import dotenv from "dotenv";

dotenv.config();

const WORKING_DIRECTORY = "./temp";


const ResearchTeamState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  team_members: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "supervisor",
  }),
  instructions: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "Solve the human's question.",
  }),
})

const llm = new ChatOpenAI({ modelName: "gpt-4o" });

const searchNode = (state: typeof ResearchTeamState.State) => {
  console.log("🚀 ~ researchNode ~ state:", state.team_members)
  const stateModifier = agentStateModifier(
    "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    [tavilyTool],
    state.team_members ?? ["Search"],
  )
  const searchAgent = createReactAgent({
    llm,
    tools: [tavilyTool],
    stateModifier,
  })
  return runAgentNode({ state, agent: searchAgent, name: "Search" });
};

const researchNode = (state: typeof ResearchTeamState.State) => {
  console.log("🚀 ~ researchNode ~ state:", state.team_members)
  const stateModifier = agentStateModifier(
    "You are a research assistant who can scrape specified urls for more detailed information using the scrapeWebpage function.",
    [scrapeWebpage],
    state.team_members ?? ["WebScraper"],
  )
  const researchAgent = createReactAgent({
    llm,
    tools: [scrapeWebpage],
    stateModifier,
  })
  return runAgentNode({ state, agent: researchAgent, name: "WebScraper" });
}

const supervisorAgent = await createTeamSupervisor(
  llm,
  "You are a supervisor tasked with managing a conversation between the" +
    " following workers:  {team_members}. Given the following user request," +
    " respond with the worker to act next. Each worker will perform a" +
    " task and respond with their results and status. When finished," +
    " respond with FINISH.\n\n" +
    " Select strategically to minimize the number of steps taken.",
  ["Search", "WebScraper"],
);

const researchGraph = new StateGraph(ResearchTeamState)
  .addNode("Search", searchNode)
  .addNode("supervisor", supervisorAgent)
  .addNode("WebScraper", researchNode)
  // Define the control flow
  .addEdge("Search", "supervisor")
  .addEdge("WebScraper", "supervisor")
  .addConditionalEdges("supervisor", (x) => x.next, {
    Search: "Search",
    WebScraper: "WebScraper",
    FINISH: END,
  })
  .addEdge(START, "supervisor");

const enterResearchChain = RunnableLambda.from(
  ({ messages }: { messages: BaseMessage[] }) => {
    return {
      messages: messages,
      team_members: ["Search", "WebScraper"],
    };
  },
);

const researchChain = enterResearchChain.pipe(() => researchGraph.compile());

const prelude = new RunnableLambda({
  func: async (state: {
    messages: BaseMessage[];
    next: string;
    instructions: string;
  }) => {
    let writtenFiles: string[] = [];
    console.log("🚀 ~ writtenFiles:", writtenFiles)
    if (
      !(await fs
        .stat(WORKING_DIRECTORY)
        .then(() => true)
        .catch(() => false))
    ) {
      await fs.mkdir(WORKING_DIRECTORY, { recursive: true });
    }
    try {
      const files = await fs.readdir(WORKING_DIRECTORY);
      for (const file of files) {
        writtenFiles.push(file);
      }
    } catch (error) {
      console.error(error);
    }
    const filesList = writtenFiles.length > 0
      ? "\nBelow are files your team has written to the directory:\n" +
        writtenFiles.map((f) => ` - ${f}`).join("\n")
      : "No files written.";
    console.log("🚀 ~ filesList:", filesList)
    return { ...state, current_files: filesList };
  },
});


// This defines the agent state for the document writing team
const DocWritingState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  team_members: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "supervisor",
  }),
  current_files: Annotation<string>({
    reducer: (x, y) => (y ? `${x}\n${y}` : x),
    default: () => "No files written.",
  }),
  instructions: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "Solve the human's question.",
  }),
})

const docWritingLlm = new ChatOpenAI({ modelName: "gpt-4o" });

const docWritingNode = (state: typeof DocWritingState.State) => {
  const stateModifier = agentStateModifier(
    `You are an expert writing a research document.\nBelow are files currently in your directory:\n${state.current_files}`,
    [writeDocumentTool, editDocumentTool, readDocumentTool],
    state.team_members ?? [],
  )
  const docWriterAgent = createReactAgent({
    llm: llm,
    tools: [writeDocumentTool, editDocumentTool, readDocumentTool],
    stateModifier,
  })
  const contextAwareDocWriterAgent = docWriterAgent.pipe(prelude);
  return runAgentNode({ state, agent: contextAwareDocWriterAgent, name: "DocWriter" });
}

const noteTakingNode = (state: typeof DocWritingState.State) => {
  console.log("🚀 ~ noteTakingNode ~ state:", state)
  const stateModifier = agentStateModifier(
    "You are an expert senior researcher tasked with writing a paper outline and" +
    ` taking notes to craft a perfect paper. ${state.current_files}`,
    [createOutlineTool, readDocumentTool],
    state.team_members ?? [],
  )
  const noteTakingAgent = createReactAgent({
    llm: docWritingLlm,
    tools: [createOutlineTool, readDocumentTool],
    stateModifier,
  })
  const contextAwareNoteTakingAgent = prelude.pipe(noteTakingAgent);
  return runAgentNode({ state, agent: contextAwareNoteTakingAgent, name: "NoteTaker" });
}

const chartGeneratingNode = async (
  state: typeof DocWritingState.State,
) => {
  console.log("🚀 ~ state:", state)
  const stateModifier = agentStateModifier(
    "You are a data viz expert tasked with generating charts for a research project." +
    `${state.current_files}`,
    [readDocumentTool, chartTool],
    state.team_members ?? [],
  )
  const chartGeneratingAgent = createReactAgent({
    llm: docWritingLlm,
    tools: [readDocumentTool, chartTool],
    stateModifier,
  })
  const contextAwareChartGeneratingAgent = prelude.pipe(chartGeneratingAgent);
  return runAgentNode({ state, agent: contextAwareChartGeneratingAgent, name: "ChartGenerator" });
}

const docTeamMembers = ["DocWriter", "NoteTaker", "ChartGenerator"];
const docWritingSupervisor = await createTeamSupervisor(
  docWritingLlm,
  "You are a supervisor tasked with managing a conversation between the" +
    " following workers:  {team_members}. Given the following user request," +
    " respond with the worker to act next. Each worker will perform a" +
    " task and respond with their results and status. When finished," +
    " respond with FINISH.\n\n" +
    " Select strategically to minimize the number of steps taken.",
  docTeamMembers,
);

// Create the graph here:
const authoringGraph = new StateGraph(DocWritingState)
  .addNode("DocWriter", docWritingNode)
  .addNode("NoteTaker", noteTakingNode)
  .addNode("ChartGenerator", chartGeneratingNode)
  .addNode("supervisor", docWritingSupervisor)
  // Add the edges that always occur
  .addEdge("DocWriter", "supervisor")
  .addEdge("NoteTaker", "supervisor")
  .addEdge("ChartGenerator", "supervisor")
  // Add the edges where routing applies
  .addConditionalEdges("supervisor", (x) => x.next, {
    DocWriter: "DocWriter",
    NoteTaker: "NoteTaker",
    ChartGenerator: "ChartGenerator",
    FINISH: END,
  })
  .addEdge(START, "supervisor");

const enterAuthoringChain = RunnableLambda.from(
  ({ messages }: { messages: BaseMessage[] }) => {
    return {
      messages: messages,
      team_members: ["DocWriter", "NoteTaker", "ChartGenerator"],
    };
  },
);

const authoringChain = enterAuthoringChain.pipe(() => authoringGraph.compile())


// let resultStream2 = await authoringChain.stream(
//   {
//     messages: [
//       new HumanMessage(
//         "Write a limerick and make a bar chart of the characters used.",
//       ),
//     ],
//   },
//   { recursionLimit: 100 },
// );

// for await (const step of resultStream2) {
//   console.log(step);
//   console.log("---");
// }


// Define the top-level State interface
const State = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "ResearchTeam",
  }),
  instructions: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "Resolve the user's request.",
  }),
});

const supervisorNode = await createTeamSupervisor(
  llm,
  "You are a supervisor tasked with managing a conversation between the" +
    " following teams: {team_members}. Given the following user request," +
    " respond with the worker to act next. Each worker will perform a" +
    " task and respond with their results and status. When finished," +
    " respond with FINISH.\n\n" +
    " Select strategically to minimize the number of steps taken.",
  ["ResearchTeam", "PaperWritingTeam"],
);

const getMessages = RunnableLambda.from((state: typeof State.State) => {
  return { messages: state.messages };
});

const joinGraph = RunnableLambda.from((response: any) => {
  return {
    messages: [response.messages[response.messages.length - 1]],
  };
});


const superGraph = new StateGraph(State)
  .addNode("ResearchTeam", async (input) => {
    const getMessagesResult = await getMessages.invoke(input);
    const researchChainResult = await researchChain.invoke({
      messages: getMessagesResult.messages,
    });
    const joinGraphResult = await joinGraph.invoke({
      messages: researchChainResult.messages,
    });
    return joinGraphResult;
  })
  .addNode("PaperWritingTeam", getMessages.pipe(authoringChain).pipe(joinGraph))
  .addNode("supervisor", supervisorNode)
  .addEdge("ResearchTeam", "supervisor")
  .addEdge("PaperWritingTeam", "supervisor")
  .addConditionalEdges("supervisor", (x) => x.next, {
    PaperWritingTeam: "PaperWritingTeam",
    ResearchTeam: "ResearchTeam",
    FINISH: END,
  })
  .addEdge(START, "supervisor");

const compiledSuperGraph = superGraph.compile();


const resultStream = compiledSuperGraph.stream(
  {
    messages: [
      new HumanMessage(
        "Look up a current event, write a poem about it, then plot a bar chart of the distribution of words therein.",
      ),
    ],
  },
  { recursionLimit: 150 },
);

for await (const step of await resultStream) {
  if (!step.__end__) {
    console.log(step);
    console.log("---");
  }
}