from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import pandas as pd
import os
from typing import Optional
class MyModel:
    def __init__(self, model_basename: str, numDataPoints: int, max_tokens: int, promptNumber: int) -> None:
        """Instantiates MyModel with Llama Model, number of datapoints, maximum tokens and prompt number.
        
        Params:
            model_basename: Llama model name to load. 
            numDataPoints: Number of data points
            max_tokens: Maximum number of tokens to be generated in a response.
            promptNumber: To store the results individually for different prompts.

        Returns:
            None.
        """
        self.model_basename = model_basename
        self.numDataPoints = numDataPoints
        self.maxTokens = max_tokens
        self.promptNumber = promptNumber
        self.model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML" # Define and initialize model_name_or_path here
        self.model_dataset = os.path.join(os.getcwd(),"../../../Datasets/","Privacy_Policy_data.xlsx")
        self.duplicate_dataset = os.path.join(os.getcwd(),"../../../Datasets/","Privacy_Policy_data_duplicate_data.xlsx")
        self.prompt=""
        print("MyModel instantiated successfully.\nPlease set the prompt using setPrompt(prompt: str) method.")

    def setPrompt(self, prompt: str) -> None:
        """This function takes a user prompt and update the model prompt.
        
        Params:
            prompt: The user prompt.

        Returns:
            None
        """
        self.prompt = prompt
        print("Prompt has been set.")

    def __loadModel(self) -> None:
        """This function instantiates the Llama Model with the provided model name.
        
        Params:
            None

        Returns:
            None

        Raises:
            ModuleNotFoundError: If provided Llama is unable to load.
        """
        try:
            model_path = hf_hub_download(repo_id=self.model_name_or_path, filename=self.model_basename)
            lcpp_llm = Llama(
                            model_path=model_path,
                            n_threads=2, # CPU cores
                            n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                            n_gpu_layers=32 # Change this value based on your model and your GPU VRAM pool.
                        )
            print("Llama model Instantiated!")
            print("GPU Layers: ",lcpp_llm.params.n_gpu_layers)
            return lcpp_llm
        except:
            raise ModuleNotFoundError("ERROR: Unable to load {} model...".format(self.model_basename))


    def __getDataPoints(self, lcpp_llm: Llama, tokens:int , numOfPoints=5) -> Optional[pd.DataFrame]:
        """This function is used to extract number of datapoints from dataset and generates response for each datapoint.
        
        Params:
            lcpp_llm: Instance of Llama Model.
            tokens: Maximum number of tokens to be generated in a response.
            numOfPoints: Number of datapoints for which the responses need to be generated.

        Returns:
            A DataFrame containing generated response text and reference text for each datapoint, or None if numOfPoints is not 30.
        """
        print(f"Checking for {numOfPoints} datapoints with {tokens} tokens...")
        df = pd.read_excel(self.model_dataset,nrows=numOfPoints)
        return self.__generateResponse(df, lcpp_llm, tokens, numOfPoints)

    def __getDuplicateDataPoints(self, lcpp_llm: Llama, tokens: int) -> Optional[pd.DataFrame]:
        """This function is used to extract all datapoints from duplicate dataset and generates response for each datapoint.
        
        Params:
            lcpp_llm: Instance of Llama Model.
            tokens: Maximum number of tokens to be generated in a response.

        Returns:
            A DataFrame containing generated response text and reference text for each datapoint, or None if numOfPoints is not 30.
        """
        print(f"Checking for duplicate datapoints with {tokens} tokens...")
        df = pd.read_excel(self.duplicate_dataset)
        self.__generateResponse(df, lcpp_llm, tokens, 0)

    def __generateResponse(self, df: pd.DataFrame, lcpp_llm: Llama, tokens: int, numOfPoints: int) -> Optional[pd.DataFrame]:
        """This function generates response with number of tokens provided for each datapoint with loaded Llama Model.
        
        Params:
            df: A DataFrame with number of datapoints.
            lcpp_llm: Instance of Llama Model.
            tokens: Maximum number of tokens to be generated in a response.
            numOfPoints: Number of datapoints for which the responses need to be generated.

        Returns:
            A DataFrame containing generated response text and reference text for each datapoint, or None if numOfPoints is not 30.
        """
        output_df = pd.DataFrame(columns=['Text', 'Reference Text', 'Generated Text', 'Generated Text Length'])
        count_processed=0
        for index, row in df.iterrows():
            prompt_template = self.prompt.format(row['Text'])
            count_processed += 1
            try:
                response = lcpp_llm(prompt=prompt_template, max_tokens=tokens, temperature=0.5, top_p=0.95,
                        repeat_penalty=1.2, top_k=150,
                        echo=True)

                # Process text in a single line
                generated_text = ' '.join(response["choices"][0]["text"].split('</INST>')[-1].strip().replace("ASSISTANT:","").split())
                reference_text = row['Reference Text']
                print("Generated_text", generated_text)
                print("Length of generated text:", len(generated_text))
                print("Reference_text:", reference_text)
                print("\n\n")

                ## Now append data to the output DataFrame
                if numOfPoints == 30:
                    output_df = pd.concat([output_df, pd.DataFrame({
                        'Text': [row['Text']],
                        'Reference Text': [row['Reference Text']],
                        'Generated Text': [generated_text],
                        'Generated Text Length': [len(generated_text)]
                    })], ignore_index=True)
            except:
                print(f"Exception: Llama response exception at {index} index.")

        if count_processed == 30 :
            return output_df

    def __saveResults(self, df: pd.DataFrame) -> None:
        """This function is used to save generated responses to file.
        
        Params:
            df: DataFrame with generated text and reference text for each datapoint.

        Returns:
            None
        
        Raises:
            TypeError: If provided param is not a DataFrame.
            Exception: Any output file creation related errors. 
        """   
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Invalid argument type. Please provide valid datatype value to preprocess.\n \
                            Requires pd.DataFrame, found " + str(type(df)))
        outputFile = self.model_basename+"_Prompt_"+str(self.promptNumber) + ".xlsx"
        outputFolder = os.path.join(os.getcwd(),f"../../../Results/{self.model_basename}")
        outputFile = os.path.join(outputFolder,outputFile)
        try:
            if not os.path.exists(outputFolder):
                os.makedirs(outputFolder)

            if os.path.exists(outputFile):
                os.remove(outputFile)

            df.to_excel(outputFile, index=False)
            print(outputFile, "Saved successfully!")

        except Exception as e:
            print("Error:", e)

    def __showResults(self,df: pd.DataFrame) -> None:
        """This function is used to show generated responses in console.
        
        Params:
            df: DataFrame with generated text and reference text for each datapoint.

        Returns:
            None
        """   
        for idx, row in df.iterrows():
            Text = row['Text']
            reference_text = row['Reference Text']
            generated_text = row['Generated Text']
            generated_text_length = row['Generated Text Length']

            print(f"Data Point {idx + 1}:")
            print(f"Text: {Text}")
            print(f"Reference Text: {reference_text}")
            print(f"Generated Text: {generated_text}")
            print(f"Length of Generated Text: {generated_text_length}\n")
            print("\n\n")


    def executeModel(self) -> None:
        """This function is used to execute Llama model, generate responses for each datapoint and store the results in a file.
        
        Params:
            None

        Returns:
            None
        """
        print("Loading the Model...")
        lcpp_llm = self.__loadModel()
        print("{} loaded successfully!!!".format(self.model_basename))
        self.__getDataPoints(lcpp_llm, tokens=2000, numOfPoints=5)
        self.__getDuplicateDataPoints(lcpp_llm, 2000)
        output_df = self.__getDataPoints(lcpp_llm, tokens=self.maxTokens,numOfPoints=self.numDataPoints)
        self.__saveResults(output_df)
        self.__showResults(output_df)
        


