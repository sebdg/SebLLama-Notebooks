echo "🚀 Creating and Installing .conda Environment 🚀"
conda create -p .conda python==3.9  tensorflow[and-cuda]  pytorch  cuda-toolkit cudatoolkit xformers -c pytorch -c nvidia -c xformers

echo "🛰️ Launching satelites 🛰️"    
conda -n .conda conda install --file requirements.txt
echo "🧨 Evening the odds 🧨"
