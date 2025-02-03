import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the topic input
st.title("Generative AI in the Medical Industry")
st.subheader("Research and Blog Generation Tool")

# User input for the topic
topic = st.text_input("Enter the topic for research and blog writing:", "Medical Industry using Generative AI")

# Define tools and agents
llm = LLM(
    model="gemini/gemini-1.5-pro-latest",
    temperature=0.1
)

#llm = LLM(
   # model="groq/llama-3.2-90b-vision-preview",
    #temperature=0.1
#)

search_tool = SerperDevTool(n=1)

senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web search",
    backstory=("You're an expert research analyst with advanced web research skills. "
               "You excel at finding, analyzing, and synthesizing information from across the internet using search tools. "
               "You're skilled at distinguishing reliable sources from unreliable ones, fact-checking, and identifying key patterns."),
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

content_writer = Agent(
    role="Content Writer",
    goal="Transform research findings into engaging blog posts while maintaining accuracy",
    backstory=("You're a skilled content writer specialized in creating engaging, accessible content from technical research. "
               "You excel at maintaining the perfect balance between informative and entertaining writing."),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

research_tasks = Task(
    description=("Conduct comprehensive research on {topic}, including:\n"
                 "- Recent developments and news\n"
                 "- Key industry trends and innovations\n"
                 "- Expert opinions and analyses\n"
                 "- Statistical data and market insights"),
    expected_output=("A detailed research report containing:\n"
                     "- Executive summary of key findings\n"
                     "- Comprehensive analysis of current trends and developments\n"
                     "- Verified facts and statistics\n"
                     "- Clear categorization of main themes and patterns"),
    agent=senior_research_analyst
)

writing_task = Task(
    description=("Using the research brief provided, create an engaging blog post that:\n"
                 "1. Transforms technical information into accessible content\n"
                 "2. Maintains all factual accuracy and citations from the research\n"
                 "3. Includes:\n"
                 "   - Attention-grabbing introduction\n"
                 "   - Well-structured body sections with clear headings\n"
                 "   - Compelling conclusion"),
    expected_output=("A polished blog post in Markdown format that:\n"
                     "- Engages readers while maintaining accuracy\n"
                     "- Contains properly structured sections\n"
                     "- Includes inline citations hyperlinked to the original source URL"),
    agent=content_writer
)

crew = Crew(
    agents=[senior_research_analyst, content_writer],
    tasks=[research_tasks, writing_task],
    verbose=True
)

# Run the app when the user clicks the button
if st.button("Generate Research & Blog"):
    with st.spinner("Generating research and blog..."):
        result = crew.kickoff(inputs={"topic": topic})
    st.success("Generation Complete!")

    # Display results
    st.subheader("Research Report")
    st.text(result["Generative AI is rapidly transforming the medical industry, promising significant improvements in patient care, drug discovery, diagnostics, and operational efficiency.  The market is experiencing exponential growth, projected to reach tens of billions of dollars within the next decade.  While the technology offers immense potential, challenges related to data privacy, ethical considerations, and regulatory frameworks need to be addressed to ensure responsible implementation."]['output'])  # Replace this with the actual report output

    st.subheader("Blog Post")
    st.text(result["Generative AI is no longer science fiction; it's rapidly transforming industries, and healthcare is at the forefront of this revolution.  Imagine a world where drug discovery is accelerated, diagnoses are more accurate, and administrative tasks are seamlessly automated. This isn't a futuristic fantasy; it's the potential of generative AI in healthcare, a market projected to reach tens of billions of dollars within the next decade ([Allied Market Research](https://www.alliedmarketresearch.com/), [Precedence Research](https://www.precedenceresearch.com/))."]['output'])  # Replace this with the actual blog output
