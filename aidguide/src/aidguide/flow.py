from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel
from .crew import AidGuideCrew

# 1. Define the State
class AidGuideState(BaseModel):
    user_input: str = ""
    user_profile: dict = {}
    is_profile_complete: bool = False
    policy_data: list = []
    final_decision: str = ""

# 2. Define the Flow
class AidGuideFlow(Flow[AidGuideState]):

    @start()
    def initialize_intake(self):
        print(f"--- Starting AidGuide Intake ---")
        # In a real app, this comes from your frontend/UI
        self.state.user_input = "I'm 65 and need help with rent in Telangana."

    @listen(initialize_intake)
    def run_screener(self):
        print(f"--- Running Screener Agent ---")
        # Calling the Screener task from your Crew
        result = AidGuideCrew().crew().kickoff(inputs={"user_input": self.state.user_input})
        
        # Logic to check if the result is JSON or a Question
        if "QUESTION" in result.raw:
            self.state.is_profile_complete = False
            return result.raw
        else:
            self.state.user_profile = result.pydantic # Assuming structured output
            self.state.is_profile_complete = True
            return "profile_complete"

    @router(run_screener)
    def route_based_on_profile(self, screener_output):
        if self.state.is_profile_complete:
            return "process_eligibility"
        else:
            return "ask_clarifying_question"

    @listen("process_eligibility")
    def execute_full_analysis(self):
        print(f"--- Profile Complete. Running Research & Judge ---")
        # This triggers the remaining sequential tasks in your crew
        final_result = AidGuideCrew().crew().kickoff(inputs=self.state.user_profile)
        self.state.final_decision = final_result.raw
        return final_result

    @listen("ask_clarifying_question")
    def handle_missing_info(self, question):
        print(f"--- Info Missing: {question} ---")
        # Logic to send this back to your Chat UI
        return question

# --- Execution ---
if __name__ == "__main__":
    flow = AidGuideFlow()
    flow.kickoff()