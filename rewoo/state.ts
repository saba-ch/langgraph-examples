import { Annotation } from "@langchain/langgraph";

export const GraphState = Annotation.Root({
  task: Annotation<string>({
    reducer: (x, y) => (y ?? x),
    default: () => "",
  }),
  planString: Annotation<string>({
    reducer: (x, y) => (y ?? x),
    default: () => "",
  }),
  steps: Annotation<{
    plan: string;
    variable: string;
    tool: string;
    toolInput: string;
  }[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  results: Annotation<Record<string, any>>({
    reducer: (x, y) => ({ ...x, ...y }),
    default: () => ({}),
  }),
  result: Annotation<string>({
    reducer: (x, y) => (y ?? x),
    default: () => "",
  }),
})