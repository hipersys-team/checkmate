#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

// Pseudo-header for TCP checksum calculation
struct pseudo_header {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint8_t placeholder;
    uint8_t protocol;
    uint16_t tcp_length;
};

struct tcp_option_swapseq {
    uint8_t type;
    uint8_t length;
    uint32_t cseq;
};

struct tcp_option_nop {
    uint8_t type;
};

// Function to calculate checksum
unsigned short checksum(void *b, int len) {    
    unsigned short *buf = b;
    unsigned int sum = 0;
    unsigned short result;

    for (sum = 0; len > 1; len -= 2)
        sum += *buf++;
    if (len == 1)
        sum += *(unsigned char *)buf;
    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    result = ~sum;
    return result;
}

int main() {
    // Change these IPs and ports as necessary
    const char *source_ip = "192.168.10.10";
    const char *dest_ip = "192.168.10.12";
    int source_port = 12345;
    int dest_port = 41000;

    // Create a raw socket
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (sock == -1) {
        perror("Socket creation failed");
        exit(1);
    }

    // Buffer for the packet
    char packet[4096];
    memset(packet, 0, sizeof(packet));

    // IP header pointer
    struct iphdr *iph = (struct iphdr *)packet;
    // TCP header pointer
    struct tcphdr *tcph = (struct tcphdr *)(packet + sizeof(struct iphdr));
    struct tcp_option_nop * nop1 = (struct tcp_option_nop *)((char*)tcph + sizeof(struct tcphdr));
    struct tcp_option_nop * nop2 = (struct tcp_option_nop *)((char*)nop1 + sizeof(struct tcp_option_nop));
    struct tcp_option_swapseq * swapseq = (struct tcp_option_swapseq *)((char*)nop2 + sizeof(struct tcp_option_nop));

    uint16_t packet_len = htons(sizeof(struct iphdr) + sizeof(struct tcphdr) + 2 * sizeof(struct tcp_option_nop) + sizeof(struct tcp_option_swapseq));

    // Fill in the IP header
    iph->ihl = 5;         // IP header length
    iph->version = 4;     // IPv4
    iph->tos = 0xC0;         // Type of service
    iph->tot_len = packet_len; // Total length (without options)
    iph->id = htons(54321); // ID of this packet
    iph->frag_off = 0;    // Fragment offset
    iph->ttl = 255;       // Time to live
    iph->protocol = IPPROTO_TCP; // Protocol (TCP)
    iph->check = 0;       // Checksum (set later)
    iph->saddr = htonl(inet_addr(source_ip)); // Source IP
    iph->daddr = htonl(inet_addr(dest_ip));   // Destination IP

    // Fill in the TCP header
    tcph->source = htons(source_port); // Source port
    tcph->dest = htons(dest_port);     // Destination port
    tcph->seq = 0;          // Sequence number
    tcph->ack_seq = 0;      // Acknowledgment number
    tcph->doff = 9;         // Data offset (TCP header size without options)
    tcph->fin = 0;
    tcph->syn = 0;          // SYN flag
    tcph->rst = 0;
    tcph->psh = 0;
    tcph->ack = 0;
    tcph->urg = 0;
    tcph->window = htons(5840); // Window size
    tcph->check = 0;        // Checksum (set later)
    tcph->urg_ptr = 0;

    nop1->type = 1;
    nop2->type = 1;

    swapseq->type = 7;
    swapseq->length = 6;
    swapseq->cseq = 0xFF00FF00;

    // Pseudo-header for checksum calculation
    struct pseudo_header psh;
    psh.src_addr = htonl(inet_addr(source_ip));
    psh.dst_addr = htonl(inet_addr(dest_ip));
    psh.placeholder = 0;
    psh.protocol = IPPROTO_TCP;
    psh.tcp_length = htons(sizeof(struct tcphdr) + 16);

    // Allocate space for TCP options (you can adjust size)
    int tcp_option_space = 8;  // Adjust this value based on how many options you want
    iph->tot_len = htons(sizeof(struct iphdr) + sizeof(struct tcphdr) + tcp_option_space);

    // Calculate IP checksum
    iph->check = checksum((unsigned short *)packet, sizeof(struct iphdr));

    // Prepare buffer for checksum calculation
    int psize = sizeof(struct pseudo_header) + sizeof(struct tcphdr) + tcp_option_space;
    char *pseudogram = malloc(psize);

    memcpy(pseudogram, (char *)&psh, sizeof(struct pseudo_header));
    memcpy(pseudogram + sizeof(struct pseudo_header), tcph, sizeof(struct tcphdr) + tcp_option_space);

    // Calculate TCP checksum
    tcph->check = checksum((unsigned short *)pseudogram, psize);

    // You can now add your custom TCP options here:
    // e.g., packet[sizeof(struct iphdr) + sizeof(struct tcphdr)] = your_custom_option_data;

    // Address to send the packet to
    struct sockaddr_in dest;
    dest.sin_family = AF_INET;
    dest.sin_port = htons(dest_port);
    dest.sin_addr.s_addr = inet_addr(dest_ip);

    // Send the packet
    if (sendto(sock, packet, ntohs(iph->tot_len), 0, (struct sockaddr *)&dest, sizeof(dest)) < 0) {
        perror("Packet sending failed");
    } else {
        printf("Packet sent successfully\n");
    }

    // Clean up
    free(pseudogram);
    close(sock);

    return 0;
}
