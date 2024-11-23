import os
import wave
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from PIL import Image
from pytesseract import pytesseract

'''This class node is defined for the min heap implementation and deque implementation.'''
class Node:
    def __init__(self,value):
        self.value = value
        self.left = self.right = None
'''This class is the node for huffman tree it includes left,right and middle for the ternary tree and character and frequency
as the data memebers.'''
class Huffman_Node:
    def __init__(self,frequency,char):
        self.left = self.right = self.middle = None
        self.character = char
        self.frequency = frequency
'''Deque is a data structure with insertion and deletion functions at both front and rear.'''
class Dequeue:
    def __init__(self):
        self.size = 100
        self.front = None
        self.rear = None
        self.length = 0
    '''This function inserts a value at the front of the dequeue at time complexity of O(1).'''
    def enqueueF(self,value):
        if(self.length==self.size):
            return "Overflow"
        new_node = Node(value)
        if(self.length==0):
            self.front = self.rear = new_node
        else:
            new_node.next = self.front
            self.front.prev = new_node
            self.front = new_node
        self.length+=1
    '''This function inserts a value at the rear of the dequeue at time complexity of O(1).'''
    def enqueue(self,value):
        if(self.length==self.size):
            return " Overflow"
        new_node = Node(value)
        if(self.length==0):
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            new_node.prev = self.rear
            self.rear = new_node
        self.length+=1
    '''This function deletes a value at the rear of the dequeue at time complexity of O(1).'''
    def dequeueR(self):
        if(self.length==0):
            return "Underflow"
        temp = self.rear
        if(self.length==1):
            self.front = self.rear = None
        else:
            temp_p = temp.prev
            temp_p.next = None
            self.rear = temp_p
        self.length-=1
        return temp.value
    '''This function deletes a value at the front of the dequeue at time complexity of O(1).'''
    def dequeue(self):
        if(self.length==0):
            return "Underflow"
        temp = self.front
        if(self.length==1):
            self.front = self.rear = None
        else:
            temp.next.prev = None
            self.front = temp.next
        self.length-=1
        return temp.value
    '''This method returns the value at the front of the dequeue at the time complexity of O(1).'''
    def first(self):
        if(self.front==None):
            return None
        return self.front.value
    '''This method returns the size of the dequeue.'''
    def len(self):
        return self.length
    '''This method returns if the dequeue is empty or not.'''
    def is_empty(self):
        if(self.length==0):
            return True
        return False
'''This is the implementation of min heap which is a priority queue whicbh extracts the minimum value in is at a time.
it is a balanced binary tree.'''
class MinHeap:
    def __init__(self):
        self.root = None
        self.rem_queue = Dequeue()
        self.length = 0
    '''This method returns the parent of a give node in the min heap using level order traversal. time complexity - O(N).'''
    def _parent(self,node):
        temp_q = Dequeue()
        temp_q.enqueue(self.root)
        while(not temp_q.is_empty()):
            temp_node = temp_q.dequeue()
            if(temp_node.left==node or temp_node.right==node):
                return temp_node
            if(temp_node.left):
                temp_q.enqueue(temp_node.left)
            if(temp_node.right):
                temp_q.enqueue(temp_node.right)
        return None
    '''returns the left child of a node at time complexity of O(1).'''
    def _left_child(self,node):
        if node.left:
            return node.left
        return None
    '''returns the right child of a node at time complexity of O(1).'''
    def _right_child(self,node):
        if node.right:
            return node.right
        return None
    '''This function swaps two nodes at time complexity O(N).'''
    def swap(self,node1,node2):
        temp = node1.value
        node1.value,node2.value = node2.value,temp
    '''This function inserts a values into the min heap using level order insertion using a dequeue and calls the swap up fucntion in which
    each values is compared with its parents frequency and swaps up so that root has smaller frequency than the child. Time complexity - O(N).'''
    def insert(self,value):
        new_node = Node(value)
        if(self.root is None):
            self.root = new_node
        else:
            t_queue  = Dequeue()
            t_queue.enqueue(self.root)
            while(not t_queue.is_empty()):
                temp = t_queue.dequeue()
                if(temp is None):
                    continue
                if(temp.left is None):
                    temp.left = new_node
                    break
                elif(temp.right is None):
                    temp.right = new_node
                    break
                t_queue.enqueue(temp.left)
                t_queue.enqueue(temp.right)
        self.length+=1
        self.rem_queue.enqueue(new_node)
        
        temp = new_node
        while(not temp==self.root):
            temp_parent = self._parent(temp)
            if(temp_parent.value.frequency<=temp.value.frequency):
                break
            self.swap(temp_parent,temp)
            temp = temp_parent
    '''This function compares a parents value with its children and swaps if the root is bigger than its children, in case both the children are smaller
    than the root it swaps with the values that is comparitively smaller values at the time complexity O(log*N)'''        
    def sink_down(self,node):
        min_node = node
        while True:
            left_c = self._left_child(node)
            right_c = self._right_child(node)
            
            if(left_c and left_c.value.frequency<min_node.value.frequency):
                min_node = left_c
            if(right_c and right_c.value.frequency<min_node.value.frequency):
                min_node = right_c
            if(min_node!=node):
                self.swap(min_node,node)
                node = min_node
            else:
                break
    '''This function replaces the roots value with the rightmost value in the child and then deletes that node. it then calls the heapify function ie 
    sink down and then returns the value that was originally at the root. Time complecity - O(log*N).'''
    def extract_minimum(self):
        if(self.root is None):
            return None
        temp = self.root
        if(temp.right is None and temp.left is None):
            self.root = None
            self.length-=1
            return temp.value
        else:
            last_node = self.rem_queue.dequeueR()
            last_node_parent = self._parent(last_node)
            
            if(last_node_parent.left==last_node):
                last_node_parent.left = None
            elif(last_node_parent.right==last_node):
                last_node_parent.right = None
            output = temp.value
            temp.value = last_node.value
            self.length-=1
        self.sink_down(self.root)
        return output
    '''This returns the size of the minimum heap at time complexity of O(N).'''
    def size(self):
        return self.length

class Huffman_Tree:
    def __init__(self):
        self.codes = dict()
        self.min_heap = MinHeap()
        self.final_node=None
    
    '''This function calculates the frequency of each character in the text by iterating through it, and stores the frquency in a dictionary
    if the character is not in the dictionary it sets the frequenct to 1 and if it already exists it adds 1 to the existing frquency value.
    Since we traverse through the entire string it takes a time complexity of O(N).'''
    def calculate_frequency(self,text):
        freq=dict()
        for char in text:   #traversing through the dictionary
            if char in freq:
                freq[char]+=1    #if character is found already once, it adds 1 to the frequncy
            else:
                freq[char]=1   #it sets frequency of a new character to 1.
        return freq     #returns the dictionary containing frquencies of characters
    
    '''This function calculates the frequency by calling the calculate_frequncy() method, while the heap's size is greater than 1 it extracts the 2
    minimum values in the heap and extracts the third value if theres is a third node if then creates a new node having frequency as the sum of 
    frquency of the three children nodes and sets the character of the node to None [this is useful for generating the code as internal nodes do not
    store any character] it then inserts the newly created node back to the min heap and continues thei process until the heap has only 1 element,
    which becomes the new root of the huffman tree. it then returns the values. this tree is thew completed huffman tree with leaf nodes as the characters
    in the original input text. This can then be used to concode the characters to ternary huffman code.'''
    def build_huffman_tree(self,text):
        
        freq_dict = self.calculate_frequency(text)
        
        for symbol, freq in freq_dict.items():
            self.min_heap.insert(Huffman_Node(freq, symbol))

        
        while self.min_heap.size() > 1:
            
            first = self.min_heap.extract_minimum()
            second = self.min_heap.extract_minimum()
            if (self.min_heap.size()>0):
                third = self.min_heap.extract_minimum()
            else:
                third = 0
            
            new_freq = first.frequency + second.frequency
            if(third):
                new_freq+=third.frequency
            
            new_node = Huffman_Node(new_freq,None)
            new_node.left = first
            new_node.middle = second
            if third:
                new_node.right = third
            
            self.min_heap.insert(new_node)
        self.final_node=self.min_heap.extract_minimum()
        return self.final_node
    '''This function encodes a text to ternary huffman codes ie. 0,1 and 2 recursively. in the huffman tree each time a node has a symbol
    ie. it is a leaf node [an actual character in the original input string] it assigns the code passed as input. for the internal node it 
    call the function for left,middle and right recursively, each time in goes left it adds 0 to the input string, for middle it adds 1 to
    the input string and for right it adds 2 to the input string. intially an empty string is passed as input and as we traverse the encoded
    string for each character is calculated. The time complexity is O(N) as we traverse through the entire huffman tree where N is the number
    of characters and 2*N-1 is the number of nodes in the huffman tree. The space complexity is O(N*log N), where N is the number of characters
    because we are using an external dictionary 'codes' and the depth of each character is log N as we traverese towards the leaf nodes.There 
    are N symbols and each symbol takes log N for insertion as we are using a min heap so the time complexity of this function O(N log N)'''
    def generate_huffman_code(self,node,curr_code):
        if node.character != None:  #if the node has a symbol
            self.codes[node.character]=curr_code   #assign the code to the character
            return 
        if node.left: #if a node has left
            self.generate_huffman_code(node.left,curr_code+"0") # add '0' to the input string as we go left
        if node.middle: #if a node has middle
            self.generate_huffman_code(node.middle,curr_code+"1") # add '1' to the input string as we go middle
        if node.right:  #if a node has right
            self.generate_huffman_code(node.right,curr_code+"2") # add '2' to the input string as we go right

    '''This function decodes a given encoded text using the Huffman tree. It traverses the tree based on the digits in the encoded text:
    '0' for left, '1' for middle, and '2' for right. Upon reaching a leaf node, it appends the corresponding character to the decoded text.
    The time complexity is O(M), where M is the length of the encoded text, as we traverse each digit once. The space complexity is O(1) 
    for the decoded string since it depends on the length of the output.'''
    def decode_huffman(self, encoded_text):
        decoded_text = ""
        current_node = self.final_node  # Use the root of the Huffman tree

        for digit in encoded_text:
            if digit == '0':
                current_node = current_node.left
            elif digit == '1':
                current_node = current_node.middle
            elif digit == '2':
                current_node = current_node.right

            # If we've reached a leaf node, append the character to the decoded text
            if current_node.character is not None:
                decoded_text += current_node.character
                current_node = self.final_node  # Reset to the root for the next character

        return decoded_text
    '''This function compresses the text by internally calling the calculate_frequency() method and builds the huffman treee
    then generates the code for each character and stores it in the static dicitonary 'codes' then it finds the compress text by 
    calling the compress_text function. This takes a time complexity of O(n + N log*N) where n is the number of characters in the input text
    and N is the number of unique characters in the input string.'''
    def compress(self,text):
        huffman_tree = self.build_huffman_tree(text) #builds the huffman tree.
        self.generate_huffman_code(huffman_tree,"") #generates the huffman codes for the characters.
        compressed_text = self.compress_text(text) #gives the compressed/encoded string.
        return compressed_text
    '''The following function is used to compress the provided text using Huffman codes
        The time--Complexity is O(N), where N is the number of characters in the provided text.
    '''
    def compress_text(self, text):
        compressed_text = ""
        for char in text: #iterates through every character in the given text.
            compressed_text += self.codes[char] #adds the compressed enocded values from the codes dictionary
        return compressed_text #returns the compresswed text
    
def audio_to_text(file_path, model_path="model"):
    if not os.path.exists(model_path):
        print("oops!!, an error occured. Try to re-execute the code")

    model = Model(model_path)

    # Convert audio to WAV format if needed
    if not file_path.endswith('.wav'):
        audio = AudioSegment.from_file(file_path)
        file_path = "converted_audio.wav"
        audio.export(file_path, format="wav")
        print("Audio converted to WAV format for processing.")

    # Open the audio file
    wf = wave.open(file_path, "rb")

    # Check if audio is mono
    if wf.getnchannels() != 1:
        print("Audio file must be mono. Please provide a mono audio file.")
        return None

    # Set up the recognizer
    rec = KaldiRecognizer(model, wf.getframerate())

    # Read audio data and perform recognition
    transcription = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            transcription += result

    # Get the final part of the transcription
    transcription += rec.FinalResult()

    # Close the audio file
    wf.close()

    print("Transcription successful!")
    return transcription

def image_to_text():
    path_to_tesseract = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    image_path = input("Enter the image path : ")
    
    image_path_new = ""
    for i in image_path:
        image_path_new+=i
        if(i=="\\"):
            image_path_new+="\\"


    img = Image.open(image_path_new)

    pytesseract.tesseract_cmd = path_to_tesseract

    text = pytesseract.image_to_string(img)
    return text

def menu():
    while(1):
        print("Welcome!!, Select the file type from the following : \n1)Audio\n2)Image\n3)Text file\n4)Exit")
        input_option = int(input("Option : "))
        
        huffman_tree = Huffman_Tree()
        if(input_option==1):
            file_path = input("Enter the path to your audio file: ")
            model_path = "D:\\HUFFMAN\\vosk-model-small-en-us-0.15"
            model_path_new = ""
            for i in model_path:
                model_path_new+=i
                if(i=="\\"):
                    model_path_new+="\\"
            result = audio_to_text(file_path, model_path_new)
            
            if result:
                encode_text = huffman_tree.compress(result)
                print("The encoded text : ",encode_text)
            else:
                print("No transcription was generated.")
        elif(input_option==2):
            text = image_to_text()
            
            if text:
                encode_text = huffman_tree.compress(text)
                print("The encoded text is : ",encode_text)
            else:
                print("No text was generated")
        elif(input_option==3):
            text_file_path = input("Enter the text File path : ")
            
            text_file_path_new = ""
            for i in text_file_path:
                text_file_path_new+=i
                if(i=="\\"):
                    text_file_path_new+="\\"
            
            fin = open(text_file_path_new,'r')
            texts=fin.read()
            if texts:
                encode_text = huffman_tree.compress(texts)
                print(encode_text)
            else:
                print("No text in the provided text file location.")
        elif(input_option==4):
            print("Thank you!!!")
            break
        else:
            print("Invalid input")
            return

        input_decode = input("\nDo you want to decode the encoded text (Y/N) :")
        if(input_decode=="Y" or input_decode=="y"):
            decoded_text = huffman_tree.decode_huffman(encode_text)
            print(decoded_text + "\nThank you!!")
        elif(input_decode=="N" or input_decode=="n"):
            print("Thank you!!!\n\n\n")
menu()
