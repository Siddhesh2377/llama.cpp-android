Okay Claude This FIle is Like a Instruction for you
So This is Llama.cpp Project and as you know this thing is heavyly optimized for desktop and other platforms, i want you to do this steps

## Step 1 
Delete All The Non-Android Code and Folders and FIles, Yes ! Delete Them Right Away
Remove Any other Backend Just Skip CPU backend, as we want only Optimized CPU backend for android, even remove valkun and open cl
Keep Every Model Compute Graphs 
Clean the Project Over all, and also completely Update the Readme, as we have cloned this so it is our property

## Step 2
Create a GGMLEngine who's Job will be to make things simple,
load/unload model via ( path & FileDeceptor For Android SAF ) 
get model complete info in Json
generate text
make as much as optimization that can be done for android device ( CPU only )

## Step 3 ToolManager
Registering tools
the current llama.cpp GBNF or system prompt tool calling failes sometimes 
so make a optimal tool-calling system which is compatable with almost every model

## Smart KV cache managment 
you decide

## Charater Engine ( vector / attention / tensor leve manupulation : requires a great research you have to do it )
Develop a Charater Engine Which is compatable almost every model with out system prompt or chattemplate 
like at the end we need heavy paramater tuning from mood to emotions to behavious, as LLM have a large amount of data
add a uncencored bool, which makes a standard model break all it's chains and gives proper uncencord out put, as this i am making this framework for Tool-Neuron which is my Offline LLM with Full privicy 
all this should be a public api so i can control via JNI

## Step 5 
Make a LLAMA-Test-CLI executable for android devices, and run every single feature i asked here to

## Context Window Tracking (Public API — expose via JNI to Kotlin)
Three metrics must be queryable at any time:
1. **Total context window size** — the model's n_ctx (max tokens the KV cache can hold)
2. **Filled context** — how many tokens are currently consumed in the KV cache
3. **Remaining context** — total minus filled
4. **Prompt fill estimate** — given the current pending prompt (before decode), estimate how many tokens it will consume and how much will remain after

All four values must be public C API functions so they can be pulled from Kotlin via JNI.

## once this is done
One all this is done and u feel that backend is ready for production after relenteless testings and improvemnt 
make edits in /home/home/AndroidStudioProjects/AiSystems/gguf_lib 
implement properly optimized JNI and Kotlin, just like a flexible SDK with all the features
write proper unit test and test every feature
ignore other modules just focus on this module

Also don't make any plans if makeing then make them in md files and fallow them don't ask for my permission as i can't give as i will be 
  asleep, you have my android device connected via adb so you can test and imprvoe as much as you want 
now i am going to sleep and handing all this to you enjoy and start building !, Bye 
