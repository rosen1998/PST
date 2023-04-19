import torch.nn as nn
import torch
import sys
from torch.autograd import Variable
import warnings

sys.path.append('../')
warnings.filterwarnings("ignore")

class LearningProcessModule(nn.Module):
    def __init__(self, seq_len, embedding_dim, device, max_position, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.device = device
        self.max_position = max_position
        self.W_1 = nn.Linear(3*embedding_dim, embedding_dim)
        self.W_2 = nn.Linear(4*embedding_dim, embedding_dim)
        self.W_3 = nn.Linear(3*embedding_dim, embedding_dim)
        self.W_4 = nn.Linear(3*embedding_dim, embedding_dim)
        self.W_5 = nn.Linear(3*embedding_dim, embedding_dim)
        self.W_6 = nn.Linear(3*embedding_dim, embedding_dim)
        self.W_7 = nn.Linear(2*embedding_dim, embedding_dim)
        self.W_8 = nn.Linear(3*embedding_dim, embedding_dim)
        self.W_9 = nn.Linear(2*embedding_dim, embedding_dim)
        self.W_10 = nn.Linear(2*embedding_dim, embedding_dim)
        self.W_11 = nn.Linear(2*embedding_dim, embedding_dim)
        self.W_12 = nn.Linear(embedding_dim, 1)
        self.W_13 = nn.Linear(embedding_dim, 1)
        self.position_encoder = nn.Embedding(self.max_position, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        torch.nn.init.xavier_uniform_(self.W_1.weight)
        torch.nn.init.xavier_uniform_(self.W_2.weight)
        torch.nn.init.xavier_uniform_(self.W_3.weight)
        torch.nn.init.xavier_uniform_(self.W_4.weight)
        torch.nn.init.xavier_uniform_(self.W_5.weight)
        torch.nn.init.xavier_uniform_(self.W_6.weight)
        torch.nn.init.xavier_uniform_(self.W_7.weight)
        torch.nn.init.xavier_uniform_(self.W_8.weight)
        torch.nn.init.xavier_uniform_(self.W_9.weight)
        torch.nn.init.xavier_uniform_(self.W_10.weight)
        torch.nn.init.xavier_uniform_(self.W_11.weight)
        torch.nn.init.xavier_uniform_(self.W_12.weight)
        torch.nn.init.xavier_uniform_(self.W_13.weight)
        torch.nn.init.xavier_uniform_(self.position_encoder.weight)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def get_answer_embedding(self, answer, embedding):
        e = torch.zeros((answer.shape[0], 2*self.embedding_dim)).to(self.device)
        gate = (answer==1).float().view(-1, 1)
        zero_embedding = torch.zeros((answer.shape[0], self.embedding_dim)).to(self.device)
        e_1 = torch.cat((embedding, zero_embedding), 1)
        e_2 = torch.cat((zero_embedding, embedding), 1)
        e = gate*e_1 + (1-gate)*e_2
        return e

    def forward(self, coding_ability, programming_knowledge, exercises, feedbacks, detail_is_ac, CIGs, CTGs, exercise_id):
        e_cig_last = torch.zeros((exercises.shape[0], self.embedding_dim)).to(self.device)
        e_f_last = torch.zeros((exercises.shape[0], self.embedding_dim)).to(self.device)
        e_e_last = torch.zeros((exercises.shape[0], self.embedding_dim)).to(self.device)
        e_id_last = torch.zeros((exercises.shape[0])).to(self.device)
        e_id_last -= 1
        pred = torch.zeros((exercises.shape[0], self.seq_len))
        pred_r = torch.zeros((exercises.shape[0], self.seq_len))
        self.coding_ability = coding_ability
        self.programming_knowledge = programming_knowledge
        position = torch.zeros((exercises.shape[0], 1)).to(self.device)
        for i in range(0, self.seq_len-1):
            position += 1
            zero_p = torch.zeros(exercises.shape[0], 1).to(self.device)
            max_p = torch.zeros(exercises.shape[0], 1).to(self.device) + self.max_position
            max_gate = (position>=self.max_position).float().view(-1, 1)
            position = max_gate*max_p + (1-max_gate)*position
            e_p = self.position_encoder(position.long()-1).view(-1, self.embedding_dim)
            e_e = exercises[:, i]
            e_f = feedbacks[:, i]
            e_cig = CIGs[:, i]
            e_ctg = CTGs[:, i]
            zero_embedding = torch.zeros((exercises.shape[0], self.embedding_dim)).to(self.device)
            e_id = exercise_id[:, i]
            e_similarity = e_id_last == e_id
            h = (e_similarity==1).float().view(-1, 1)
            e_s = torch.cat((e_cig, e_e, e_f), 1)
            e_c = torch.cat((e_cig_last, e_ctg, e_e, e_f_last), 1)
            submission = self.dropout(self.tanh(self.W_1(e_s)))
            change = self.dropout(self.tanh(self.W_2(e_c)))
            coding_info = torch.cat((h*change+(1-h)*zero_embedding, (h*zero_embedding+(1-h)*submission)), 1)
            coding_ability_hat = self.tanh(self.W_3(torch.cat((self.coding_ability, coding_info), 1)))
            forget_gate_ca = self.sig(self.W_4(torch.cat((self.coding_ability, coding_info), dim=1)))
            input_gate_ca = self.sig(self.W_5(torch.cat((self.coding_ability, coding_info), dim=1)))
            self.coding_ability = forget_gate_ca*self.coding_ability+input_gate_ca*coding_ability_hat
            e_id_next = exercise_id[:, i+1]
            e_similarity_next = e_id_next == e_id
            h_next = (e_similarity_next==1).float().view(-1, 1)
            final_solution = self.tanh(self.W_6(e_s))
            forget_gate_pk = self.sig(self.W_7(torch.cat((self.programming_knowledge, final_solution), dim=1)))
            input_gate_pk = self.sig(self.W_8(torch.cat((self.programming_knowledge, final_solution, e_p), dim=1)))
            LG = self.tanh(self.W_9(torch.cat((self.programming_knowledge, final_solution), 1)))
            self.programming_knowledge = h_next*self.programming_knowledge + (1-h_next)*(forget_gate_pk*self.programming_knowledge+input_gate_pk*LG)
            e_e_next = exercises[:, i+1]
            solution = self.relu(self.W_10(torch.cat((self.programming_knowledge, e_e_next), dim=1)))
            y = self.relu(self.W_11(torch.cat((self.coding_ability, solution), dim=1)))
            next_pred = self.sig(self.W_12(y))
            r = self.sig(self.W_13(y))
            pred[:, i+1] = torch.squeeze(next_pred)
            pred_r[:, i+1] = torch.squeeze(r)
            e_id_last = e_id
            e_f_last = e_f
            e_e_last = e_e
            e_cig_last = h_next*e_cig + (1-h_next)*zero_embedding
            position = h_next*position + (1-h_next)*zero_p
        e_e = exercises[:, self.seq_len-1]
        e_ctg = CTGs[:, self.seq_len-1]
        e_cig = CIGs[:, self.seq_len-1]
        return pred, pred_r

class PST(nn.Module):
    def __init__(self, seq_len, num_exercises, embedding_dim, device, max_position, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_exercises = num_exercises
        self.embedding_dim = embedding_dim
        self.device = device
        self.max_position = max_position
        self.dropout = dropout
        self.W_1 = nn.Linear(2*self.embedding_dim, self.embedding_dim)
        self.W_2 = nn.Linear(2*self.embedding_dim, self.embedding_dim)
        torch.nn.init.xavier_uniform_(self.W_1.weight)
        torch.nn.init.xavier_uniform_(self.W_2.weight)
        self.learning_fitting_encoder = LearningProcessModule(self.seq_len, self.embedding_dim, self.device, self.max_position, self.dropout)
        self.exercise_encoder = nn.Embedding(self.num_exercises, self.embedding_dim)
        self.tanh = nn.Tanh()
        torch.nn.init.xavier_uniform_(self.exercise_encoder.weight)
    
    def encode_feedback(self, answer):
        one = torch.ones((answer.shape[0], self.embedding_dim)).to(self.device)
        zero = torch.zeros((answer.shape[0], self.embedding_dim)).to(self.device)
        e_a = answer.long()*one + (1-answer.long())*zero
        return e_a

    def forward(self, detail_is_ac, exercises, e_cig, e_ctg):
        self.programming_knowledge = Variable(torch.zeros(detail_is_ac.shape[0], self.embedding_dim).to(self.device))
        self.coding_ability = Variable(torch.zeros(detail_is_ac.shape[0], self.embedding_dim).to(self.device))
        e_cig = self.W_1(e_cig)
        e_ctg = self.W_2(e_ctg)
        detail_is_ac = detail_is_ac.view(-1, 1)
        e_f = self.encode_feedback(detail_is_ac.long())
        e_f = e_f.view(-1, self.seq_len, self.embedding_dim)
        e_e = self.exercise_encoder(exercises)
        pred, pred_r = self.learning_fitting_encoder(self.coding_ability, self.programming_knowledge, e_e, e_f, detail_is_ac, e_cig, e_ctg, exercises)
        return pred, pred_r