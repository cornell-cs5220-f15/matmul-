import sys
import subprocess

def generate(args):
    if len(args) < 4:
        return
    fin = open(args[0], "r")
    fout = open("dgemm_" + args[1] + ".c", "w")
    fill = args[4:]

    for l in fin:
        for i, f in enumerate(fill):
            l = l.replace("$" + str(i) + "$", f)
        fout.write(l)
    fin.close()
    fout.close()

    fin = open(args[2], "r")
    fout = open("job-" + args[3] + ".pbs", "w")
    for l in fin:
        for i, f in enumerate(fill):
            l = l.replace("$" + str(i) + "$", f)
        fout.write(l)
    fin.close()
    fout.close();

def build(args):
    subprocess.call(["module", "load", "cs5220", "&&", "make", "matmul-" + args[0]])

def run(args):
    subprocess.call(["module", "load", "cs5220", "&&", "make", "run",  "timing-" + args[0] + ".csv"])

def main():
    if len(sys.argv) < 2:
        print "Usage: python mmgen.py <command>"
        print "    generate <file> dgemm_<output> [inputs...]"
        print "    build matmul-<file>"
        print "    run timing-<file>"
    else:
        target = sys.argv[1]
        args = sys.argv[2:]
        #print sys.argv[1]
        #print sys.argv[2:]
        if target == "generate":
            generate(args)

main()
    

