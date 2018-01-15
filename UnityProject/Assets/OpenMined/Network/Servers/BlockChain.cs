using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;
using System.IO;
using Newtonsoft.Json;

namespace OpenMined.Network.Servers
{
    public class BlockChain : MonoBehaviour
    {
        private Coroutine routine;
 
        public void Start()
        {
            var o = ReadConfig();

            if (o["trainer"].ToObject<bool>())
            {
                Debug.Log("POLLING");
                PollNext();
            }
        }

        void PollNext()
        {
            this.routine = StartCoroutine(PollNetwork());
        }

        IEnumerator PollNetwork()
        {
            var request = new Request();

            yield return request.GetBlockNumber(this);
            yield return request.GetModel(this);

            Debug.Log("Blockchain polled");

            yield return new WaitForSeconds(10);
            PollNext();
        }
        
        JObject ReadConfig()
        {
            using (StreamReader reader = File.OpenText("Assets/OpenMined/Config/config.json"))
            {
                return (JObject)JToken.ReadFrom(new JsonTextReader(reader));
            }
        }
    }
}