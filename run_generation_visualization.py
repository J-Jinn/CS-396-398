"""
CS-396 Senior Project I
Project: Huggingface-Transformers GPT2 Text Modeling and Prediction
Advisor: Professor Kenneth Arnold
Coordinator: Professor Keith VanderLinden
Author: Joseph Jinn

run_generation_custom.py defines and implements a bare-bones text modeling and prediction program using the GPT2
    model and tokenizer.

#########################################################################################

Notes:

https://github.com/dunovank/jupyter-themes
(Jupyter Notebook Themes)
https://towardsdatascience.com/bringing-the-best-out-of-jupyter-notebooks-for-data-science-f0871519ca29
(useful additions for Jupyter Notebook)
https://medium.com/@rbmsingh/making-jupyter-dark-mode-great-5adaedd814db
(Jupyter dark-mode settings; my eyes are no longer bleeding...)
https://github.com/ipython-contrib/jupyter_contrib_nbextensions
(Jupyter extensions)
https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
(PyTorch tutorial on character-level RNN)

Enter this in Terminal (for use with jupyter-themes):
jt -t monokai -f fira -fs 13 -nf ptsans -nfs 11 -N -kl -cursw 5 -cursc r -cellw 95% -T

#########################################################################################

Important files to reference:

modeling_gpt2.py
 - The GPT2 model source code.

tokenization_gpy2.py
 - The tokenizer class for the GPT2 model.

#########################################################################################

Reference Material to understand the Theoretical Foundation of GPT2:
https://en.wikipedia.org/wiki/Language_model
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

It would also be helpful to have some concept about beam search… I’m not super-happy with what my Googling obtains but…
https://en.wikipedia.org/wiki/Beam_search
https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

Also maybe helpful but don’t get distracted:
the first 20 minutes or so of this (everything after that is details of training, skip it.)
https://www.youtube.com/watch?v=Keqep_PKrY8
https://medium.com/syncedreview/language-model-a-survey-of-the-state-of-the-art-technology-64d1a2e5a466
https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

#########################################################################################

More Notes:

- CTRL + M + L (while in command mode): Adds code cell line numbers (very useful for debugging)

- Select code fragment --> right-click --> Execute selection in Python console (Alt + Shift + E)
    - executes selected (highlighted) code without re-running entire file.

- CTRL + Q (brings up API documentation in Pycharm)

- CTRL + Space (brings up list of functions)

- Shift + Escape (close API documentation panel)
"""

#########################################################################################
#########################################################################################

# Import required packages and libraries.
import torch  # PyTorch.
import pandas as pd # Pandas.
from tqdm import trange  # Instantly make your loops show a smart progress meter.
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#########################################################################################

# Load the GPT2-model.
model_class = GPT2LMHeadModel  # Specifies the model to use.
tokenizer_class = GPT2Tokenizer  # Specifies the tokenizer to use for the model.
tokenizer = tokenizer_class.from_pretrained('gpt2')  # Use pre-trained model.
model = model_class.from_pretrained('gpt2')  # User pre-trained model.
model.to('cpu')  # Specifies what machine to run the model on.
model.eval()  # Specifies that the model is NOT in training mode.

#########################################################################################


# noinspection DuplicatedCode,PyUnresolvedReferences
def extract_top_k_tokens(filtered_logits, k_value):
    """
    This function utilizes the torch.topk() function to choose the "k" most likely words.

    torch.topk performs a similar function to Softmax and argmax.
    Uses the words' "scores" to choose the top "k" most likely predicted words (tokens).

    - torch.topk
     - Returns the :attr:`k` largest elements of the given :attr:`input` tensor along a given dimension.

    Non-statistical and probabilistic method, so results are deterministic (always the same).

    Parameters:
        filtered_logits - entire vocabulary with assigned scores from GPT2 model.
        k_value - choose "k" top most likely words.

    Return:
        my_topk - top "k" word tokens as Tensors.
    """
    topk_debug = False

    # Return the top "k" most likely (highest score value) words in sorted order..
    my_topk = torch.topk(filtered_logits, k=k_value, dim=1, sorted=True)
    if topk_debug:
        print(f"My torch.topk object: {my_topk}\n")
        print(f"torch.topk indices: {my_topk.indices}")
        print(f"torch.topk values: {my_topk.values}\n")

    # https://stackoverflow.com/questions/34750268/extracting-the-top-k-value-indices-from-a-1-d-tensor
    # https://stackoverflow.com/questions/53903373/convert-pytorch-tensor-to-python-list

    # Indices = encoded words, Values = scores.
    if topk_debug:
        print(f"\nDecoded torch.topk indices: {[tokenizer.decode(idx) for idx in my_topk.indices.squeeze().tolist()]}")
        print(f"\nDecoded torch.topk values: {tokenizer.decode(my_topk.indices.squeeze().tolist())}\n")

        print(f"topk indices shape: {my_topk.indices.shape}")
        print(f"topk indices shape after squeeze: {my_topk.indices.squeeze().shape}")
        print(f"topk indices after squeeze: {my_topk.indices.squeeze()}\n")

        # https://stackoverflow.com/questions/43328632/pytorch-reshape-tensor-dimension

        print(f"topk indices 1st element in Tensor: {my_topk.indices[0][0]}")
        print(f"topk indices 1st element in Tensor shape: {my_topk.indices[0][0].shape}")
        print(f"topk indices 1st element in Tensor with added dimension: {my_topk.indices[0][0].unsqueeze(0)}")
        print(f"topk indices 1st element in Tensor with added dimension shape: "
              f"{my_topk.indices[0][0].unsqueeze(0).shape}\n")

    if topk_debug:
        # Ghetto looping through topk indices.
        for elements in my_topk.indices[0]:
            if topk_debug:
                print(f"topk word: {elements}")
                print(f"topk word shape: {elements.shape}")
                print(f"topk word shape after un-squeezing: {elements.unsqueeze(0).unsqueeze(0).shape}")

            # Set each element as the next token for text prediction and generation.
            next_token = elements.unsqueeze(0).unsqueeze(0)
            if topk_debug:
                print(f"Next token shape: {next_token.shape}")
                print(f"Next token: {next_token}")
                print(f"Decoded next token(s): {tokenizer.decode(next_token.squeeze().tolist())}\n")

    # Returns the Tensor array of the top "k" word tokens
    return my_topk


#########################################################################################

# noinspection DuplicatedCode,PyUnresolvedReferences
def prediction_generation(context_tokens, generated, prediction_option):
    """
    This function makes text prediction using the GPT2 model and outputs the results.

    Parameters:
       context_tokens - the encoded raw text string.
       generated - context_tokens wrapped as a PyTorch Tensor.
       prediction_option - 'auto' or 'interactive' text prediction option.
    """
    import random  # Random number generator.

    temperature = 1  # Default value.
    iterations = 20  # Default value.
    k_value = 3  # Top "k" words to choose.

    if prediction_option == "interactive":
        valid = False
        while not valid:
            print(f"\nNote: To terminate the program, enter 'exit' for any of the requested inputs.  "
                  f"Once all inputs have received a value, the program will then terminate.")

            print(f"\nNote: Temperature is a hyper-parameter of LSTMs (and neural networks generally) used to control "
                  f"the randomness of predictions by scaling the logits before applying softmax.")
            temperature = input(f"Set temperature value to (real number > 0): ")

            print(f"\nNote: This controls how many iterations to generate the top 'k' most likely word tokens based on "
                  f"the preceding token, which controls the # of word tokens the predicted text will consist of.")
            iterations = input(f"Set the number of text prediction iterations for current string to: (integer > 0): ")

            print(f"\nNote: This controls the # of tokens returned by the torch.topk() greedy sampling function.")
            k_value = input(f"Enter the 'k' value for top 'k' most likely word token generation (integer > 0): ")

            if temperature == "exit" or iterations == "exit" or k_value == "exit":
                print(f"Terminating program...")
                quit(0)

            try:
                if float(temperature) > 0.0 and int(iterations) > 0 and int(k_value) > 0:
                    valid = True
                else:
                    print(f"Invalid value(s) detected! Please choose valid value(s)!\n")
            except TypeError:
                continue

    generated_array = []  # List of "generated" PyTorch Tensor containing encoded word tokens.
    token_score_array = []  # List of "scores" for each token in the current iteration of topk greedy sampling.
    alternative_route = []  # List containing the alternative choices we could have made.

    logits_debug = False
    topk_debug = False
    output_debug = False
    alternative_debug = False

    # Create list of PyTorch Tensors containing encoded original raw text string.
    # Create a list of word token score values initially set to 1.0.
    for i in range(0, int(k_value)):
        generated_array.append(generated)
        token_score_array.append(1.)

    chosen_generated = generated_array[0]  # For initial iteration.

    # Setup for displaying alternative routes.
    for element in range(0, int(k_value)):
        alternative_route.append(tokenizer.decode(context_tokens))
    if alternative_debug:
        print(f"")
        for element in alternative_route:
            print(f"Contents of alternative_route nested lists: {element}")

    ############################################################################################

    with torch.no_grad():  # This specifies not to use stochastic gradient descent!
        for _ in trange(int(iterations)):

            ############################################################################################

            # Note: Feeding the results back into the model is the beginnings of a beam search algorithm.
            # Currently, randomly chooses one of the "generated" Tensors to feed back in.
            if logits_debug:
                print(f"Original generated shape: {generated}")
                print(f"Generated array element 0 shape: {generated_array[0]}")
                print(f"token_score_array element 0 shape: {token_score_array[0]}\n")

            if prediction_option == "auto":
                chosen_generated = generated_array[random.randint(0, int(k_value) - 1)]

            # Call to GPT2 model generates a Tensor object containing "scores" for the entire vocabulary.
            outputs = model(input_ids=chosen_generated)
            if logits_debug:
                print(f"Outputs shape: {list(outputs)[0].shape}\n")
                print(f"Outputs: {list(outputs)[0]}\n")  # Outputs is a tensor containing a lot of stuff...

            next_token_logits = outputs[0][:, -1, :] / (float(temperature) if float(temperature) > 0 else 1.)
            if logits_debug:
                print(f"Next token logits shape: {next_token_logits.shape}\n")
                print(f"Next token logits: {next_token_logits}\n")

            filtered_logits = next_token_logits  # Set to default name from run_generation.py

            ############################################################################################

            # Call function to extract the top "k" word tokens based on their scores.
            my_topk = extract_top_k_tokens(filtered_logits, int(k_value))

            if prediction_option == "auto":
                # Ghetto looping through topk indices.
                counter = 0
                for elements in my_topk.indices[0]:
                    if topk_debug:
                        print(f"topk word: {elements}")
                        print(f"topk word shape: {elements.shape}")
                        print(f"topk word shape after un-squeezing: {elements.unsqueeze(0).unsqueeze(0).shape}")

                    # Set each element as the next token for text prediction and generation.
                    next_token = elements.unsqueeze(0).unsqueeze(0)
                    if topk_debug:
                        print(f"Next token shape: {next_token.shape}")
                        print(f"Next token: {next_token}")
                        print(f"Decoded next token(s): {tokenizer.decode(next_token.squeeze().tolist())}\n")

                    # Concatenate the chosen token (predicted word) to the end of the tokenized (encoded) string.
                    # Then, add to the array of "generated" PyTorch tensors by modifying the original generated.
                    generated_array[counter] = (torch.cat((chosen_generated, next_token), dim=1))
                    if topk_debug:
                        print(f"Generated shape: {chosen_generated.shape}")
                        print(f"Generated: {chosen_generated}")
                        print(f"Decoded 'generated' tokens: {tokenizer.decode(chosen_generated.squeeze().tolist())}\n")

                    counter += 1

                ############################################################################################

                # Output the text prediction results.
                print(f"\n###############################################################################")
                print(f"Note: The '#' at the beginning and end delimit the start and end of the text.")
                print(f"Original (excluding text prediction) raw text string: {tokenizer.decode(context_tokens)}\n")
                counter = 0
                for gen in generated_array:
                    out = gen
                    if output_debug:
                        print(f"Contents of 'out': {out}")

                    # This line removes the original text but keeps appending the generated words one-by-one
                    # (based on iteration length).
                    out = out[:, len(context_tokens):].tolist()
                    if output_debug:
                        print(f"Contents of 'out' after .tolist(): {out}\n")
                        print(f"Length of context tokens:{len(context_tokens)}\n")

                    # Outputs the result of the text modeling and prediction.
                    for o in out:
                        # Decode - convert from token ID's back into English words.
                        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                        #     text = text[: text.find(args.stop_token) if args.stop_token else None]
                        print(f"Prediction {counter} of {int(k_value - 1)} for this iteration based on previous "
                              f"iterations' randomly selected tokens (using RNG).")
                        print(f"Predicted (excluding original raw input text) text string: #{text}#")
                    counter += 1
                print(f"###############################################################################\n")

            ############################################################################################

            if prediction_option == "interactive":
                chosen = False
                while not chosen:
                    print(f"\nEnter 'exit program' or 'Exit Program' to terminate the program.")
                    print(f"The top k={k_value} tokens are:")
                    print(f"Note: The '#' are there to delimit the start and end of the token since tokens "
                          f"can include '\\n' and other invisible characters.")
                    print(f"Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.\n")
                    counter = 0
                    for elements in my_topk.indices[0]:
                        print(f"Token {counter}: #{tokenizer.decode(elements.unsqueeze(0).tolist())}#")
                        # print(f"{type(tokenizer.decode(elements.unsqueeze(0).tolist()))}")
                        alternative_route[counter] = alternative_route[counter] + tokenizer.decode(
                            elements.unsqueeze(0).tolist())
                        counter += 1

                    choose_token = input(f"\nChoose a token to use for the next iteration of text prediction:")

                    if choose_token == "exit program" or choose_token == "Exit Program":
                        print(f"Terminating program...")
                        quit(0)

                    for elements in my_topk.indices[0]:
                        if choose_token == str(tokenizer.decode(elements.unsqueeze(0).tolist())):
                            next_token = elements.unsqueeze(0).unsqueeze(0)
                            chosen_generated = (torch.cat((chosen_generated, next_token), dim=1))
                            chosen = True
                            break

                ############################################################################################

                # Output the text prediction results.
                print(f"\n###############################################################################")
                print(f"Original (excluding text prediction) raw text string: {tokenizer.decode(context_tokens)}\n")

                print(f"All routes that user could have made in choosing a token from the current iteration:")
                counter = 0
                for i in range(0, int(k_value)):
                    print(f"Route {counter}: {alternative_route[counter]}")
                    counter += 1

                out = chosen_generated
                if output_debug:
                    print(f"Contents of 'out': {out}")

                # This line removes the original text but keeps appending the generated words one-by-one (based on
                # iteration length).
                out = out[:, len(context_tokens):].tolist()
                if output_debug:
                    print(f"Contents of 'out' after .tolist(): {out}\n")
                    print(f"Length of context tokens:{len(context_tokens)}\n")

                # Outputs the result of the text modeling and prediction.
                for o in out:
                    # Decode - convert from token ID's back into English words.
                    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                    #     text = text[: text.find(args.stop_token) if args.stop_token else None]
                    print(f"Note: The '#' at the beginning and end delimit the start and end of the text.")
                    print(f"Predicted (excluding original raw input text) text string: #{text}#")
                    print(f"###############################################################################\n")

            # Updated setup for displaying alternative routes based on each iteration's chosen token.
            # This essentially means we display alternative routes based on the previously chosen token(s).
            counter = 0
            for element in range(0, int(k_value)):
                alternative_route[counter] = (tokenizer.decode(context_tokens) + text)
                counter += 1

            ############################################################################################

            # Store the scores for each token.
            counter = 0
            for elements in my_topk.values[0]:
                token_score_array[counter] = elements.unsqueeze(0).unsqueeze(0)
                if topk_debug:
                    print(f"topk word score: {elements}")
                    print(f"topk word score shape: {elements.shape}")
                    print(f"topk word score shape after un-squeezing: {elements.unsqueeze(0).unsqueeze(0).shape}")
                counter += 1

#########################################################################################


def main():
    """
    Main encodes the raw text string, wraps in PyTorch Tensor, and calls prediction_generation().
    Executes forever until user enters "exit" or "Exit".

    Parameters: None
    Return: None
    """
    main_debug = False
    context_debug = False
    num_samples = 1  # Default value.
    user_option = "auto"

    repeat_query = True
    while repeat_query:
        user_option = input(f"Type 'auto' or 'interactive'.")

        if user_option == "exit" or user_option == "Exit":
            return
        elif user_option != "auto" and user_option != "interactive":
            repeat_query = True
            print(f"Unrecognized option - type 'auto' or 'interactive'!")
        else:
            repeat_query = False

    ############################################################################################

    while True:
        raw_text = ""
        while len(raw_text) == 0:
            raw_text = input("Enter a string: ")
            if len(raw_text) == 0:
                print(f"Please enter something that is NOT a empty string!")

        # Quit the program.
        if raw_text == "exit" or raw_text == "Exit":
            print(f"Terminating program execution.")
            break

        # Encode raw text.
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        if main_debug:
            print(f"Raw text: {raw_text}\n")
            print(f"Context tokens: {context_tokens}\n")

        context = context_tokens  # Set to name as in run_generation.py

        # Convert to a PyTorch Tensor object (numpy array).
        context = torch.tensor(context, dtype=torch.long, device='cpu')
        if context_debug:
            print(f"Context shape: {context.shape}")
            print(f"Context converted to PyTorch Tensor object: {context}\n")

        # Un-squeeze adds a dimension to the Tensor array.
        # Repeat adds x-dimensions and repeats the Tensor elements y-times.
        context = context.unsqueeze(0).repeat(num_samples, 1)
        if context_debug:
            print(f"Context shape after 'un-squeeze': {context.shape}")
            print(f"Context after 'un-squeeze': {context}\n")

        generated = context  # Set to name as in run_generation.py

        # Generate and output text prediction results.
        prediction_generation(context_tokens, generated, user_option)
        print(f"Iterations for current string has ended.  Will request user enter new string.\n")


#########################################################################################

# Execute the program.
# Select below and Run with "Alt + Shift + E" to avoid re-running entire fire and re-loading model every-time.
if __name__ == '__main__':
    main()

############################################################################################
