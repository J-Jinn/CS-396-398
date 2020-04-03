"""
CS-398 Senior Project II
Project: Huggingface-Transformers GPT2 Text Modeling and Prediction
Advisor: Professor Kenneth Arnold
Coordinator: Professor Keith VanderLinden
Author: Joseph Jinn

run_generation_console_program.py defines and implements a bare-bones text modeling and prediction program using the GPT2
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

https://stackoverflow.com/questions/34750268/extracting-the-top-k-value-indices-from-a-1-d-tensor
https://stackoverflow.com/questions/53903373/convert-pytorch-tensor-to-python-list
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

# FIXME - need to associate chosen tokens for predicted sentence with token selected choice dictionary.

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

    # Return the top "k" most likely (highest score value) words in sorted order..
    my_topk = torch.topk(filtered_logits, k=k_value, dim=1, sorted=True)

    # Returns the Tensor array of the top "k" word tokens
    return my_topk


#########################################################################################

# noinspection DuplicatedCode,PyUnresolvedReferences
def prediction_generation(context_tokens, generated):
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

    generated_array = []  # List of "generated" PyTorch Tensor containing encoded word tokens.
    token_score_array = []  # List of "scores" for each token in the current iteration of topk greedy sampling.

    logits_debug = False
    topk_debug = False
    output_debug = True
    json_debug = True

    # Create list of PyTorch Tensors containing encoded original raw text string.
    # Create a list of word token score values initially set to 1.0.
    for i in range(0, int(k_value)):
        generated_array.append(generated)
        token_score_array.append(1.)

    ############################################################################################

    # Data structure to store all token lists from every iteration.
    token_options_all_lists = {}

    # Data structure to store all tokens for each iteration.
    token_options_list = []

    iteration_counter = 0

    with torch.no_grad():  # This specifies not to use stochastic gradient descent!
        for increament in trange(int(iterations)):

            # Note: Feeding the results back into the model is the beginnings of a beam search algorithm.
            # Currently, randomly chooses one of the "generated" Tensors to feed back in.
            if logits_debug:
                print(f"Original generated shape: {generated}")
                print(f"token_score_array element 0 shape: {token_score_array[0]}\n")

            # Randomly select the previously generated text to use to predict the next word token(s).
            chosen_generated = generated_array[random.randint(0, int(k_value) - 1)]
            # chosen_generated = generated_array[k_value - 1]
            if logits_debug:
                print(f"Chosen generated: {chosen_generated}")

            # Append each list of token for each iteration of predictions.
            chosen_generated_to_list = chosen_generated[0].tolist()
            chosen_generated_to_list_last_element = chosen_generated_to_list[len(chosen_generated_to_list) - 1]
            chosen_generated_to_list_last_element_decoded = tokenizer.decode(chosen_generated_to_list_last_element)

            if json_debug:
                print(f"chosen generated to list: {chosen_generated_to_list}")
                print(f"chosen generated to list last element: {chosen_generated_to_list_last_element}")
                print(f"chosen generated to list last element decoded: {chosen_generated_to_list_last_element_decoded}")

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
                    print(f"Decoded next token(s): {tokenizer.decode(next_token.squeeze().tolist())}")
                    print(f"Decoded next token(s) data type: "
                          f"{type(tokenizer.decode(next_token.squeeze().tolist()))}\n")

                # Append each token to the list for the current iteration through the topk words.
                token_options_list.append(tokenizer.decode(next_token.squeeze().tolist()))

                # Concatenate the chosen token (predicted word) to the end of the tokenized (encoded) string.
                # Then, add to the array of "generated" PyTorch tensors by modifying the original generated.
                generated_array[counter] = (torch.cat((chosen_generated, next_token), dim=1))
                if topk_debug:
                    print(f"Generated shape: {chosen_generated.shape}")
                    print(f"Generated: {chosen_generated}")
                    print(f"Decoded 'generated' tokens: {tokenizer.decode(chosen_generated.squeeze().tolist())}\n")

                counter += 1

            if topk_debug:
                print(f"Token options list: {token_options_list}")

            iteration_counter += 1

            # if increment > 0:
            #     token_options_all_lists[chosen_generated_to_list_last_element_decoded] = token_options_list
            #     print(f"All token options list stored in dictionary {token_options_all_lists}")

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
                    print(f"Contents of 'out' after .tolist(): {out}")
                    print(f"Length of context tokens:{len(context_tokens)}")
                    test = out[0][-1]
                    print(f"test: {test}")
                    print(f"Decoded test: {tokenizer.decode(test)}")
                    print(f"Decoded out: {tokenizer.decode(out[0])}")

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

            # Store chosen token and associated choices of tokens for each iteration.
            select = tokenizer.decode(out[0][-1], clean_up_tokenization_spaces=True)
            token_options_all_lists[str(select)] = token_options_list
            print(f"All token options list stored in dictionary {token_options_all_lists}")
            token_options_list = []

            complete_string = f"{tokenizer.decode(context_tokens)}{text}"
            print(f"Complete string: {complete_string}")

            # Store the scores for each token.
            counter = 0
            for elements in my_topk.values[0]:
                token_score_array[counter] = elements.unsqueeze(0).unsqueeze(0)
                if topk_debug:
                    print(f"topk word score: {elements}")
                    print(f"topk word score shape: {elements.shape}")
                    print(f"topk word score shape after un-squeezing: {elements.unsqueeze(0).unsqueeze(0).shape}")
                counter += 1

    # print(f"All token options list stored in dictionary {token_options_all_lists}")

    # Return the original string and predicted text following it, as well as all the possible options.
    print([complete_string, token_options_all_lists])
    return [complete_string, token_options_all_lists]


#########################################################################################

# noinspection DuplicatedCode
def main(user_input_string):
    """
    Main encodes the raw text string, wraps in PyTorch Tensor, and calls prediction_generation().
    Executes forever until user enters "exit" or "Exit".

    Parameters: The user text input.
    Return: The predicted text.
    """
    num_samples = 1  # Default value.

    # Encode raw text.
    context_tokens = tokenizer.encode(user_input_string, add_special_tokens=False)

    context = context_tokens  # Set to name as in run_generation.py

    # Convert to a PyTorch Tensor object (numpy array).
    context = torch.tensor(context, dtype=torch.long, device='cpu')

    # Un-squeeze adds a dimension to the Tensor array.
    # Repeat adds x-dimensions and repeats the Tensor elements y-times.
    context = context.unsqueeze(0).repeat(num_samples, 1)

    generated = context  # Set to name as in run_generation.py

    # Generate and output text prediction results.
    return prediction_generation(context_tokens, generated)


#########################################################################################

# Execute the program.
# Select below and Run with "Alt + Shift + E" to avoid re-running entire fire and re-loading model every-time.
# Note: you need to run the entire code-base at least once with Alt + Shift + E to have everything in memory.
# Note: anytime you make changes to any function, need to re-run entire function for the changes to implement.
if __name__ == '__main__':
    main("Hello")

############################################################################################
