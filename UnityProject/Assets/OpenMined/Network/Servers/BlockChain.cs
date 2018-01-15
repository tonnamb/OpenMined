using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System;

namespace OpenMined.Network.Servers
{
    public class BlockChain : MonoBehaviour
    {
        private Coroutine routine;
 
        public void Start()
        {
            PollNext();
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
    }
}