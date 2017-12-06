import re
def find(edup,mergp):
    edus = open(edup).readlines()
    edus = [e.strip("\n").strip().split(" ") for e in edus]

    text = open(mergp).readlines()
    text = [e.strip("\n").strip().split("\t") for e in text]
    text = [t for t in text if t[0]]
    n_line = 0
    print(len(edus),len(text))
    counter = len(edus[0])
    for t in text:
    
        n_edu = int(t[-1])
        if counter>0:
            counter -= 1
            if n_line != n_edu-1:
                return n_edu
        else:
            n_line += 1
            counter = len(edus[n_line])
            if counter>0:
                counter -=1
                if n_line != n_edu-1:
                    return n_edu


def fix_edu(mergep,start_idx):
        text_ = []
        text = open(mergep).readlines()
        text = [t.strip("\n").split("\t") for t in text]
        text = [t for t in text if len(t)>0 and t[0]]
        for t in text:
            if int(t[-1])>start_idx:
                t[-1]=str(int(t[-1])-1)
                t[0]=str(int(t[0])-1)
                text_.append(("\t").join(t))
            
            if int(t[1])==1:
                text_.insert(-1,"\n")
        new_text = ("\n").join(text_)
        new_text = re.sub(r"\n\n\n","\n\n",new_text)
        return new_text

def rewrite(mergp,text,pos):
    old_text = open(mergp).readlines()
    old_text = [t.strip("\n").split("\t") for t in old_text]
    old_text = [t for t in old_text if len(t)>0 and t[0]]
    
    text = text.split("\n")
    text = [t.strip("\n").split("\t") for t in text]
    text = [t for t in text if len(t)>0 and t[0]]
    
    new_text = []
    last_idx = 1
    for t in old_text:
        if int(t[-1])>pos:
            break
        new_text.append(("\t").join(t))
        last_idx += 1
        if int(t[1])==1:
            last_idx = 1
            new_text.insert(-1,"\n")
    
    first = True
    for t in text:
        if int(t[1])==1:
            first=False
            new_text.append("\n")

        if first:
            t[1]=str(int(t[1])+last_idx)
        
        new_text.append(("\t").join(t))
        
        
    new_text = ("\n").join(new_text)
    new_text = re.sub(r"\n\n\n","\n\n",new_text)
    print(new_text)
    f = open(mergp,"w")
    f.write(new_text)
    


if __name__=="__main__":
    edup = "data/test/wsj_1376.out.edus"
    mergp= "data/test/wsj_1376.out.merge"

    pos = find(edup,mergp)

    print("pos",pos)
    while pos:
        text = fix_edu(mergp,pos-1)
        rewrite(mergp,text,pos-1)
        pos = find(edup,mergp)
        print("pos",pos)