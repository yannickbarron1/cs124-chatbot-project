# PA7, CS124, Stanford
# v.1.0.4
#
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util

import numpy as np
import re
import porter_stemmer


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        self.name = 'movierecommender'

        self.stemmer = porter_stemmer.PorterStemmer()

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')


        # sang - read movies.txt
        with open('data/movies.txt', encoding="utf8") as f:
            lines = f.readlines()
        # sang - pull out titles, each element is a string 'title1 (title2) (title3) ... (year)'
        string_title = []
        for i in range(len(lines)):
            string_title.append(re.split('%',lines[i])[1])
        self.string_title = string_title
        # sang - change each element into a list [title1, title2, title3, ..., year]
        # it also takes care of articles
        list_title = []
        for i in range(len(string_title)):
            a = re.split('\s\(',string_title[i])
            ## drop unnecessary parts
            for i in range(1,len(a)):
                a[i]=a[i].replace('a.k.a. ','')
                a[i]=a[i].replace(')','')
            for i in range(0,len(a)):
                if re.match(r', [A-Z][a-z][a-z]',a[i][-5:]):
                    b = a[i][-5:]
                    a[i]=a[i].replace(b,'')
                    b = b.replace(', ','')
                    a[i]= b + ' ' + a[i]        
                elif re.match(r', [A-Z][a-z]',a[i][-4:]) or re.match(r", L'",a[i][-4:]):
                    b = a[i][-4:]
                    a[i]=a[i].replace(b,'')
                    b = b.replace(', ','')
                    a[i]= b + ' ' + a[i]
                elif re.match(r', [A-Z]',a[i][-3:]):
                    b = a[i][-3:]
                    a[i]=a[i].replace(b,'')
                    b = b.replace(', ','')
                    a[i]= b + ' ' + a[i]               
            list_title.append(a)
        self.list_title = list_title
        # sang - change each list into a single string 'title1 year title2 year ...'
        # so that it can be compared to input
        longstring_title = []
        for i in range(len(list_title)):
            title = list_title[i][0] + ' ' + list_title[i][-1]
            for j in range(1,len(list_title[i])-1):
                title = title + ' ' + list_title[i][j] + ' ' + list_title[i][-1]
            longstring_title.append(title)
        self.longstring_title = longstring_title

        # rachel - read sentiment.txt
        with open('data/sentiment.txt', encoding="utf8") as f:
            sentiment_line = f.readlines()
        # rachel - pull sentiment, each line in the format of "token,neg/pos" and store in dict
        # at the same time, change "pos" to 1 and "neg" to -1
        token_sentiment = {}
        for i in range(len(sentiment_line)):
            split = sentiment_line[i].split(",")
            sent_value = 0
            if split[1] == "pos\n":
                sent_value = 1
            elif split[1] == "neg\n":
                sent_value = -1
            token_sentiment[split[0]] = sent_value
        self.token_sentiment = token_sentiment
        # rachel - create regexes for extract sentiment
        self.neg_words_regex = r"\b[a-zA-Z]*(?:not|never|no|n't|ain't)\s+\b\w+\b"
        self.punc_trans_regex = r"(?:and|but|because|since|in that|however|although|yet|despite|in spite of|even though|though|on the other hand|then again|nevertheless|nonetheless|instead|otherwise|notwithstanding|even so|\.|,|;|:|\!|\?)"
        self.special_tokens = {
            r"enjoy(?:ed|s)?": "enjoy",
            r"fanc(?:ies|ied)": "fancy"
        }
        self.strong_pos_tokens = [r"re+a+lly+", r"extre+mely+", r"ve+r+y+", r"lo+ved+?",
                              r"hi+ghly+", "entirely", "exceptionally", r"extraordinar(?:y|ily)", 
                              r"total(ly+)?", r"absolu+te(ly+)?", "quite", r"definite(ly)?", 
                              "undoubtedly", r"strong+(ly+)?", r"!+", r"lo+t", r"gre+a+t"]
        self.strong_neg_tokens = [r"\bn(ever|ot)\b", r"\bwors[et]?[s]?\b", r"\b(?:awful|terrible|dreadful|horrible|abominable)\b",
                                r"\b(?:suck[s]?|sucks|sucky|nasty|shitty|crappy|atrocious|rotten|lousy)\b",
                                r"\b(?:disgusting|repulsive|revolting|vile|foul|offensive|grotesque)\b",
                                r"\b(?:hate[s]?|hated|hateful|detestable|loathsome|odious)\b",
                                r"\b(?:abhor[s]?|abhorred|abhorrent)\b",
                                r"\b(?:horrendous|appalling|deplorable|execrable|ghastly)\b",
                                r"\b(?:repugnant|hideous|unspeakable|unbearable|unacceptable)\b",
                                r"\b(?:insufferable|obnoxious|invidious|repellant)\b",
                                "at all"]
        # rachel - list of words that indicate a turn in a sentence
        self.turn_tokens = ['but', 'however', 'nevertheless', 'nonetheless']
                        

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = Chatbot.binarize(ratings)

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Nice to meet you!"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        goodbye_message = "Thank you for having me with you! Have a wonderful day."
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:

        # this is the creative section

        # this is the main subsection of the creative section         
            if self.sentiment_counter<5 and self.spellcheck==[] and self.multiple_movies_found==[]:
                extracted_titles = self.extract_titles(line)
                if not extracted_titles:
                    response = "I want to hear about movies. Please tell me about a movie you've seen."
                elif len(extracted_titles) > 1:
                    response = "Please tell me about one movie at a time. Go ahead."
                else:
                    valid_title = extracted_titles[0]
                    matches = self.find_movies_by_title(valid_title)
                    if not matches:
                        self.spellcheck = self.find_movies_closest_to_title(valid_title)
                        if not self.spellcheck:
                            response = "Are you sure you entered the correct title? Please rewrite what you said with the correct title."
                        else:
                            print("I don't think there's a movie titled \"{}\".".format(valid_title))
                            response = 'Did you mean \"{}\"? (yes/no)'.format(self.list_title[self.spellcheck[0]][0])
                            self.saved_line = line
                    elif len(matches) > 1:
                        response = "There are multiple movies titled \"{}\". Which one of these did you mean?".format(valid_title)
                        self.multiple_movies_found = matches
                        self.saved_line = line
                        for i in matches:
                            print(self.string_title[i])
                    else:
                        extracted_sentiment = self.extract_sentiment_starter(line)
                        ## need to change to extract_sentiment when creative part is done #################################################
                        if not extracted_sentiment:
                            response = "I'm sorry, I'm not sure if you liked \"{}\". Tell me more about it.".format(valid_title)
                        else:
                            self.user_ratings[matches[0]] = extracted_sentiment
                            self.sentiment_counter += 1 # needs to reset after 5 valid inputs, self.user_ratings does not reset
                            #test
                            if self.sentiment_counter < 5:
                                if extracted_sentiment == 1:
                                    response = "So you enjoyed \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                                elif extracted_sentiment == -1:
                                    response = "So you didn't enjoy \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                            elif self.sentiment_counter == 5:
                                if extracted_sentiment == 1:
                                    print("So you enjoyed \"{}\".".format(valid_title))
                                elif extracted_sentiment == -1:
                                    print("So you didn't enjoy \"{}\".".format(valid_title))
                                self.recommended_movies = self.recommend(self.user_ratings, self.ratings)
                                self.just_recommended = self.recommended_movies[0]
                                response = "I think you'll like \"{}\". Do you want another recommendation? (yes/no)".format(self.string_title[self.just_recommended])
            
            # this subsection takes care of typos, mostly copied from main subsection
            elif self.sentiment_counter<5 and self.spellcheck!=[]:
                if line == 'no':
                    response = "Are you sure you entered the correct title? Please rewrite what you said with the correct title."
                    self.spellcheck = []
                    self.saved_line = ''                
                elif line == 'yes':
                    valid_title = self.list_title[self.spellcheck[0]][0]
                    self.spellcheck = []
                    print("So you meant \"{}\".".format(valid_title))
                    matches = self.find_movies_by_title(valid_title)
                    if len(matches) > 1:
                        response = "There are multiple movies titled \"{}\". Which one of these did you mean?".format(valid_title)
                        self.multiple_movies_found = matches
                        for i in matches:
                            print(self.string_title[i])
                    else:
                        extracted_sentiment = self.extract_sentiment_starter(self.saved_line)
                        ## need to change to extract_sentiment when creative part is done #################################################
                        if not extracted_sentiment:
                            response = "I'm sorry, I'm not sure if you liked \"{}\". Tell me more about it.".format(valid_title)
                        else:
                            self.user_ratings[matches[0]] = extracted_sentiment
                            self.sentiment_counter += 1 # needs to reset after 5 valid inputs, self.user_ratings does not reset
                            #test
                            if self.sentiment_counter < 5:
                                if extracted_sentiment == 1:
                                    response = "So you enjoyed \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                                elif extracted_sentiment == -1:
                                    response = "So you didn't enjoy \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                            elif self.sentiment_counter == 5:
                                if extracted_sentiment == 1:
                                    print("So you enjoyed \"{}\".".format(valid_title))
                                elif extracted_sentiment == -1:
                                    print("So you didn't enjoy \"{}\".".format(valid_title))
                                self.recommended_movies = self.recommend(self.user_ratings, self.ratings)
                                self.just_recommended = self.recommended_movies[0]
                                response = "I think you'll like \"{}\". Do you want another recommendation? (yes/no)".format(self.string_title[self.just_recommended])
            
                else:
                    response = "I'm so sorry but I don't understand what you're trying to say. Please tell me if I have the right title by saying either yes or no."

            # this subsection takes care of when there are multiple movies found under one title, mostly copied from main subsection
            elif self.sentiment_counter<5 and self.multiple_movies_found!=[]:
                matches = self.disambiguate(line,self.multiple_movies_found)
                if not matches:
                    response = "I'm so sorry but I don't understand what you're trying to say. Please indicate which one of these movies you meant."
                else:
                    self.multiple_movies_found=[]
                    valid_title = self.string_title[matches[0]]
                    print("So you meant \"{}\".".format(valid_title))
                    extracted_sentiment = self.extract_sentiment_starter(self.saved_line)
                    ## need to change to extract_sentiment when creative part is done #################################################
                    if not extracted_sentiment:
                        response = "I'm sorry, I'm not sure if you liked \"{}\". Tell me more about it.".format(valid_title)
                    else:
                        self.user_ratings[matches[0]] = extracted_sentiment
                        self.sentiment_counter += 1 # needs to reset after 5 valid inputs, self.user_ratings does not reset
                        #test
                        if self.sentiment_counter < 5:
                            if extracted_sentiment == 1:
                                response = "So you enjoyed \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                            elif extracted_sentiment == -1:
                                response = "So you didn't enjoy \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                        elif self.sentiment_counter == 5:
                            if extracted_sentiment == 1:
                                print("So you enjoyed \"{}\".".format(valid_title))
                            elif extracted_sentiment == -1:
                                print("So you didn't enjoy \"{}\".".format(valid_title))
                            self.recommended_movies = self.recommend(self.user_ratings, self.ratings)
                            self.just_recommended = self.recommended_movies[0]
                            response = "I think you'll like \"{}\". Do you want another recommendation? (yes/no)".format(self.string_title[self.just_recommended])
        

            elif self.sentiment_counter==5:
                if line == 'yes' and self.just_recommended != self.recommended_movies[-1]:
                    for i in range(len(self.recommended_movies)-1):
                        if self.just_recommended == self.recommended_movies[i]:
                            self.just_recommended = self.recommended_movies[i+1]
                            if i<len(self.recommended_movies)-2:
                                response = "I think you'll also like \"{}\". Do you want another recommendation? (yes/no)".format(self.string_title[self.just_recommended])
                            elif i == len(self.recommended_movies)-2:
                                response = "I think you'll also like \"{}\". This is my final recommendation. If you want more, please tell me about other movies you liked or didn't like!".format(self.string_title[self.just_recommended])
                                self.sentiment_counter = 0
                elif line == 'no':
                    response = 'Thank you for talking to me today. Have a wonderful day!'
                    self.sentiment_counter = 0
                else:
                    response = "I'm so sorry but I don't understand what you're trying to say. Please tell me if you want another recommendation by saying either yes or no."





        # this is the starter section
        else:
            if self.sentiment_counter<5:
                extracted_titles = self.extract_titles(line)
                if not extracted_titles:
                    response = "I want to hear about movies. Please tell me about a movie you've seen."
                elif len(extracted_titles) > 1:
                    response = "Please tell me about one movie at a time. Go ahead."
                else:
                    valid_title = extracted_titles[0]
                    matches = self.find_movies_by_title(valid_title)
                    if not matches:
                        response = "I don't think there's a movie titled \"{}\", sorry...".format(valid_title)
                    elif len(matches) > 1:
                        response = "There are multiple movies titled \"{}\". Could you be more specific?".format(valid_title)
                    else:
                        extracted_sentiment = self.extract_sentiment(line)
                        if not extracted_sentiment:
                            response = "I'm sorry, I'm not sure if you liked \"{}\". Tell me more about it.".format(valid_title)
                        else:
                            self.user_ratings[matches[0]] = extracted_sentiment
                            self.sentiment_counter += 1 # needs to reset after 5 valid inputs, self.user_ratings does not reset
                            #test
                            if self.sentiment_counter < 5:
                                if extracted_sentiment == 1:
                                    response = "So you enjoyed \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                                elif extracted_sentiment == -1:
                                    response = "So you didn't enjoy \"{}\". Tell me about another movie you liked or didn't like.".format(valid_title)
                            elif self.sentiment_counter == 5:
                                if extracted_sentiment == 1:
                                    print("So you enjoyed \"{}\".".format(valid_title))
                                elif extracted_sentiment == -1:
                                    print("So you didn't enjoy \"{}\".".format(valid_title))
                                self.recommended_movies = self.recommend(self.user_ratings, self.ratings)
                                self.just_recommended = self.recommended_movies[0]
                                response = "I think you'll like \"{}\". Do you want another recommendation? (yes/no)".format(self.string_title[self.just_recommended])
                                
            elif self.sentiment_counter==5:
                if line == 'yes' and self.just_recommended != self.recommended_movies[-1]:
                    for i in range(len(self.recommended_movies)-1):
                        if self.just_recommended == self.recommended_movies[i]:
                            self.just_recommended = self.recommended_movies[i+1]
                            if i<len(self.recommended_movies)-2:
                                response = "I think you'll also like \"{}\". Do you want another recommendation? (yes/no)".format(self.string_title[self.just_recommended])
                            elif i == len(self.recommended_movies)-2:
                                response = "I think you'll also like \"{}\". This is my final recommendation. If you want more, please tell me about other movies you liked or didn't like!".format(self.string_title[self.just_recommended])
                                self.sentiment_counter = 0
                elif line == 'no':
                    response = 'Thank you for talking to me today. Have a wonderful day!'
                    self.sentiment_counter = 0
                else:
                    response = "I'm so sorry but I don't understand what you're trying to say. Please tell me if you want another recommendation by saying either yes or no."



        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response


    @staticmethod
    def preprocess(text):
        """Optioanl: Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        return text


    def extract_titles_starter(self, text):
        # sang - basic title extraction function for starter mode
        # input is a string, output is a list of strings
        titles = re.findall(r'"(.*?)"', text)
        return titles
            

    def create_phrases(self, text):
        # sang - this function creates all possible phrases from the input/title so that they can be compared to each other
        # intput is a string, output is two lists, whose elements are strings
        # list of phrases without elimination of punctuation and changing upper > lower case
        a = re.split('\s',text)
        phrases1 = []
        for i in range(len(a)):
            for j in range(i,len(a)):
                phrase = a[i]
                for k in range(i+1,j+1):
                    phrase = phrase + ' ' + a[k]
                phrases1.append(phrase)
                
        # list of phrases with elimination of punctuation and changing upper > lower case
        a = re.sub(r'[^\w\s]', '',text)
        a = a.lower()
        a = re.split('\s',a)
        phrases2 = []
        for i in range(len(a)):
            for j in range(i,len(a)):
                phrase = a[i]
                for k in range(i+1,j+1):
                    phrase = phrase + ' ' + a[k]
                phrases2.append(phrase)
        return phrases1, phrases2
    

    def compare_input_to_movies(self, text):
        # sang - this function finds best-matching movies based on input
        # intput is a string, output is a list of lists whose format is
        # [matched string (raw), matched string (raw), matched string (processed), movie title, movie index]
        
        # this part finds all possible matches
        matches=[]
        inputo, inputp = self.create_phrases(text)
        for i in range(len(inputp)):
            for j in range(len(self.longstring_title)):
                compareo, comparep = self.create_phrases(self.longstring_title[j])
                if inputp[i] in comparep:
                    matches.append([inputo[i],inputo[i],inputp[i],self.string_title[j],j])
        if len(matches)==0:
            best_matches = []
        # sang - this part finds best matches based on length of overlapping phrase
        else:
            match_length = []
            for i in range(len(matches)):
                length = len(matches[i][2])
                match_length.append(length)

            match_length = np.array(match_length)
            max_index = np.argwhere(match_length == np.max(match_length))
            max_index = np.squeeze(max_index)
            max_index = max_index.tolist()

            if isinstance(max_index,int):
                best_matches = [matches[max_index]]
            elif isinstance(max_index, list):
                best_matches = []
                for i in max_index:
                    best_matches.append(matches[i])
        
        return best_matches


    def extract_titles_creative(self, text):
        # sang - this function finds the part in the input that seems most likely to be a movie title
        # input is a string, output is a list of strings
        titles = []
        a = re.sub(r'[^\w\s]', '',text)
        a = a.lower()
        words_input = a.split()
        if '"' in text:
            titles = self.extract_titles_starter(text)
        else:
            for i in range(len(self.list_title)):
                for k in range(len(self.list_title[i])-1):
                    b = re.sub(r'[^\w\s]', '',self.list_title[i][k])
                    b = b.lower()
                    words_movie = b.split()
                    for j in range(len(words_input)-len(words_movie)+1):
                        if words_input[j:j+len(words_movie)] == words_movie:
                            titles.append(words_movie)
            for i in range(len(titles)):
                concat = titles[i][0]
                for j in range(1,len(titles[i])):
                    concat = concat + ' ' + titles[i][j]
                titles[i]=concat
        return titles


    def extract_titles(self, text):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        if not self.creative:
            titles = self.extract_titles_starter(text)
        elif self.creative:
            titles = self.extract_titles_creative(text)
        return titles


    def find_movies_by_title_starter(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        articles = ["An", "A", "The"] 
        contains_article = False
        contains_year = False
        title_regex = ""

        title_tokens = title.split()

        if title_tokens[0] in articles: 
            contains_article = True
        if re.match(r"\([0-9]+\)", title_tokens[-1]): 
            contains_year = True

        if contains_article and contains_year:
            title_regex = re.escape(" ".join(title_tokens[1:-1]) + ", " 
                                    + title_tokens[0] + " " + title_tokens[-1])
        elif contains_article:
            title_regex = " ".join(title_tokens[1:]) + ", " + title_tokens[0] + r" \([0-9]+\)"
        elif contains_year:
            title_regex = re.escape(title)
        else:
            title_regex = title + r" \([0-9]+\)"
        
        matches = []

        for i in range(len(self.titles)):
            if re.match(title_regex, self.titles[i][0]):
                matches.append(i)

        return matches

        
    def find_movies_by_title_creative(self, title):
        # sang - intput is a string, output is a LIST of movie indices
        movies_found = []
        
        a = self.compare_input_to_movies(title)
        if a==[]:
            movies_found = []
        else:
            for j in range(len(a)):
                movies_found.append(a[j][4])
        if len(movies_found)==0:
            movies_found = []
        return movies_found


    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        if not self.creative:
            movies = self.find_movies_by_title_starter(title)
        elif self.creative:
            movies = self.find_movies_by_title_creative(title)
        return movies


    def negation_handling(self, text):
        """rachel - finds negation sequences in a given input

        Parameters:
        text: a prepocessed text that's ripped of movie titles 
        Returns:
        neg_pairs: a list of sequences starting with each occurrence of a negation word
        """
        matches = re.findall(self.neg_words_regex + r"(?:(?!_)\W*\b\w+\b)*?", text)
        neg_words_dict = {}
        for match in matches:
            substring = re.findall(match + r"\s?(.*?)(?=" + self.punc_trans_regex + "|$)", text)
            if substring:
                negated_words = match + ' ' + ' '.join(re.findall(r"\b\w+\b", substring[0]))
                neg_words_dict[match] = negated_words
        return list(neg_words_dict.values())

    
    def extract_sentiment_starter(self, preprocessed_input):
        """rachel - Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 
        0 if the sentiment of the text is neutral (no sentiment detected), 
        or +1 if the sentiment of the text is positive. ---

        Use the sentiment lexicon to process the sentiment. ---

        Negative handling?? (see Ed post) ---

        Edge cases:
        1. neutral sentiment --- 
        2. when movie titles contain sentiment "I hate Love Affair" ---
        3. when one clause is more important than the other - not sure???
        4. need to account for lack of ending punctuation ---
        5. when two occurrences of the same description is present
        6. rest of the the list of special tokens to be added 
        7. when a negation word finds another negation word
        """
        final_score = 0
        # get rid of movie titles in the input
        titles = self.extract_titles(preprocessed_input)
        for title in titles:
            preprocessed_input = preprocessed_input.replace(title, "")
        input = preprocessed_input.lower()
        # acquire a list of negation sequences, each starting with one negation word
        neg_seq = self.negation_handling(input)
        # go through all negation sequences by looking at the starting negation word
        for sequence in neg_seq:
            sequence_tokens = sequence.split()
            # go through each token after the 0 index, check their sentiment, and record to results
            for token in sequence_tokens[1:]:
                for pattern, replacement in self.special_tokens.items():
                    if re.search(pattern, token):
                        stemmed_token = replacement
                        break
                    else:
                        stemmed_token = self.stemmer.stem(token)
                # update score if there is sentiment (mind that this is negative sentiment)
                if stemmed_token in self.token_sentiment:
                    final_score -= int(self.token_sentiment[stemmed_token])
        # delete negation sequences from original input
        for sequence in neg_seq:
            if sequence.endswith(' '):
                sequence = sequence.rstrip()
            input = input.replace(sequence, "")
        # go through the rest of the input and update sentiment score
        remaining_tokens = input.split()
        for token in remaining_tokens:
            for pattern, replacement in self.special_tokens.items():
                    if re.search(pattern, token):
                        stemmed_token = replacement
                        break
                    else:
                        stemmed_token = self.stemmer.stem(token)
            # update score if there is sentiment
            if stemmed_token in self.token_sentiment:
                final_score += int(self.token_sentiment[stemmed_token])  
        # return the sentiment
        if final_score >= 1:
            return 1
        elif final_score == 0:
            return 0
        else:
            return -1


    def extract_sentiment_creative(self, preprocessed_input):
        """rachel - As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        implementations: 
        - strong words
        """
        final_score = 0
        # get rid of movie titles in the input
        titles = self.extract_titles(preprocessed_input)
        for title in titles:
            preprocessed_input = preprocessed_input.replace(title, "")
        input = preprocessed_input.lower()
        # acquire a list of negation sequences, each starting with one negation word
        neg_seq = self.negation_handling(input)
        # go through all negation sequences by looking at the starting negation word
        for sequence in neg_seq:
            sequence_tokens = sequence.split()
            # go through each token after the 0 index, check their sentiment, and record to results
            for token in sequence_tokens[1:]:
                for pattern, replacement in self.special_tokens.items():
                    if re.search(pattern, token):
                        stemmed_token = replacement
                        break
                    else:
                        stemmed_token = self.stemmer.stem(token)
                # update score if there is sentiment (mind that this is negative sentiment)
                if stemmed_token in self.token_sentiment:
                    final_score -= int(self.token_sentiment[stemmed_token])
        # delete negation sequences from original input
        for sequence in neg_seq:
            if sequence.endswith(' '):
                sequence = sequence.rstrip()
            input = input.replace(sequence, "")
        # go through the rest of the input and update sentiment score
        remaining_tokens = input.split()
        for token in remaining_tokens:
            for pattern, replacement in self.special_tokens.items():
                    if re.search(pattern, token):
                        stemmed_token = replacement
                        break
                    else:
                        stemmed_token = self.stemmer.stem(token)
            # update score if there is sentiment
            if stemmed_token in self.token_sentiment:
                final_score += int(self.token_sentiment[stemmed_token])  
        # return the sentiment
        if final_score >= 1:
            return 1
        elif final_score == 0:
            return 0
        else:
            return -1
    

    def extract_sentiment(self, preprocessed_input):
        """rachel - this function combines the two functions

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        score = 0
        if not self.creative:
            score = self.extract_sentiment_starter(preprocessed_input)
        elif self.creative:
            score = self.extract_sentiment_creative(preprocessed_input)
        return score


    def extract_sentiment_for_movies(self, preprocessed_input):
        """rachel - Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        Do not have to worry about missing quotation marks or incorrect capitalization

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        

    def get_minimum_edit_distance(self, str1, str2, max_distance):
        """Maeghan: calculates minimum edit distance between two words. """
        len1 = len(str1) + 1
        len2 = len(str2) + 1

        D = np.zeros((len1, len2))

        for i in range(len1):
            D[i][0] = i

        for j in range(len2):
            D[0][j] = j

        for i in range(1, len1):
            for j in range(1, len2):
                cost = 0 if str1[i - 1].lower() == str2[j - 1].lower() else 2
                D[i][j] = np.min([D[i-1][j] + 1, 
                                    D[i][j-1] + 1, 
                                    D[i-1][j-1] + cost])
                
                if all(x > max_distance for x in D[i, :]):
                   return max_distance + 1
                
        return D[len1 - 1, len2 - 1]
    
    def format_title_to_movie_standard(self, title):
        """Maeghan: calculates minimum edit distance between two words. """
        articles = ["An", "A", "The"]
        contains_article = False
        contains_year = False

        title_tokens = title.split()

        if title_tokens[0] in articles: 
            contains_article = True
        if re.match(r"\([0-9]+\)", title_tokens[-1]): 
            contains_year = True

        if contains_article and contains_year:
            processed_title = " ".join(title_tokens[1:-1]) + ", " + title_tokens[0] + " " + title_tokens[-1]
        elif contains_article:
            processed_title = " ".join(title_tokens[1:]) + ", " + title_tokens[0]
        else:
            processed_title = title

        return processed_title, contains_year

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        edit_distances = []

        formatted_title, contains_year = self.format_title_to_movie_standard(title)

        for i in range(len(self.titles)):
            movie_title = self.titles[i][0] if contains_year else " ".join(self.titles[i][0].split()[:-1])
            edit_distances.append(self.get_minimum_edit_distance(formatted_title,  
                                                            movie_title, 
                                                 max_distance))
        
        min_distance = min(edit_distances)
        matches = [] if min_distance > max_distance else [i for i, 
                                                          v in enumerate(edit_distances) 
                                                          if v ==  min_distance]
        return matches

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movie
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        results = []

        clarification_regex = rf'^{clarification}$|^{clarification}\W|\W{clarification}$|\W{clarification}\W'

        for candidate_index in candidates:
            if re.search(clarification_regex, self.titles[candidate_index][0], re.IGNORECASE):
                results.append(candidate_index)

        return results

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        # tmp has binarized values, but 0's are also incorrectly set to -1.
        # This is fixed in the second np.where statement
        tmp = np.where(ratings > threshold, 1, -1)
        binarized_ratings = np.where(ratings != 0, tmp, 0)
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        uDotV = np.dot(u, v)
        normu = np.linalg.norm(u)
        normv = np.linalg.norm(v)
        norm_prod = normu*normv
        if norm_prod == 0:  #avoids division by 0
            return 0        #sets similarity to 0 indicating that the movies were not rated by any of the same people
        similarity = uDotV/(norm_prod)
        return similarity
    
    # builds similarity vector using the current movie as the reference
    def find_similarities(self, curr_movie, ratings_matrix):
        similarities = np.apply_along_axis(self.similarity, 1, ratings_matrix, curr_movie)
        return similarities

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """
        #Populate this list with k movie indices to recommend to the user.
        idxes = np.nonzero(user_ratings)
        rated = ratings_matrix[idxes]
        similarities = np.apply_along_axis(self.find_similarities, 1, rated, ratings_matrix) #has a nested np.apply_along_axis to avoid for loops
        weighted = np.dot(user_ratings[idxes], similarities)
        weighted[idxes] = -1000000  #hardcoded but it forces all movies that were already rated to show up at the beginning of the vector
        recommendations = list(np.argsort(weighted)[-1: -k - 1: -1])
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        return """
        The movie recommender is a chatbot that recommends movie titles to the user based
        on user's profile. In the basic mode, the chatbot requires 5 datapoints of input from
        the user to decide what movie to recommend to the user. The inputs from the user are
        generally strict, such as movies having to be quoted in quotation marks, as well as 
        the required simplicity of the sentence. In the creative mode, the chatbot is able 
        to perform more nuanced lingusitic tasks and allows for more verbal freedom from the
        user. The chatbot is a first attempt to mimick human capacity of natural language 
        processing and understanding.
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
